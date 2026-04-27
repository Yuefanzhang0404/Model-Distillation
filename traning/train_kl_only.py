import json
import logging
import math
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class DistillationConfig: #configuration for distillation training
    teacher_model: str = "Qwen/Qwen2.5-3B-Instruct" #teacher model
    student_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct" #student model
    mode: str = "kl_only"

    data_path: Optional[str] = "data/distill_25k/mixed_25k.jsonl"
    data_size: int = 25000
    max_seq_length: int = 512

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 3
    learning_rate: float = 5e-5

    temperature: float = 2.0
    alpha: float = 0.0
    beta: float = 1.0
    gamma: float = 0.0

    output_dir: str = "runs/qwen3b_smol360m_kl_only"
    save_steps: int = 500
    logging_steps: int = 50
    resume_from_checkpoint: Optional[str] = None

    device: str = "cuda:0"
    use_bf16: bool = True
    use_gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"
    load_teacher_in_4bit: bool = True

    num_workers: int = 2
    pin_memory: bool = True
    grad_clip: float = 1.0

# Define VocabProjector a simple linear layer 
class VocabProjector(nn.Module):
    def __init__(self, student_hidden_size: int, teacher_vocab_size: int):
        super().__init__()
        self.projector = nn.Linear(student_hidden_size, teacher_vocab_size)
        nn.init.normal_(self.projector.weight, std=0.02)

        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projector(hidden_states)

#Extract the final answer from the teacher's response
class DistillationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: str,
        data_size: int,
        max_seq_length: int,
    ):
        if data_path is None:
            raise ValueError("A real data_path is required.")

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.texts = self._load_texts(Path(data_path), data_size)

        if not self.texts:
            raise ValueError(f"No valid training samples found in {data_path}")

    #process data 
    def _load_texts(self, path: Path, limit: int) -> List[str]:
        files = list(path.glob("*.jsonl")) if path.is_dir() else [path]
        texts: List[str] = []

        for file_path in files:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = self._parse_line(line)

                    if text is not None:
                        texts.append(text)

                    if len(texts) >= limit:
                        return texts

        return texts
    
    #handle json 
    @staticmethod
    def _parse_line(line: str) -> Optional[str]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            return None

        text = (
            item.get("text")
            or item.get("content")
            or f"{item.get('prompt', '')}\n{item.get('response', '')}"
        ).strip()

        return text if text else None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

#create batch
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }

#Load the teacher and student models
def setup_logger(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(output_dir)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(Path(output_dir) / "training.log")
    stream_handler = logging.StreamHandler(sys.stdout)

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def get_device(config: DistillationConfig) -> torch.device:
    return torch.device(config.device if torch.cuda.is_available() else "cpu")


def get_amp_dtype(config: DistillationConfig) -> torch.dtype:
    if config.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def get_device_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":")[1])

    return 0


def freeze_model(model: nn.Module) -> None:
    model.eval()

    for param in model.parameters():
        param.requires_grad = False


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

#record loss values and training speed
def average_records(
    records: List[Dict[str, float]],
    epoch: int,
    global_step: int,
    mode: str,
    step_key: str = "global_step",
) -> Dict[str, Any]:
    if not records:
        raise ValueError("Cannot average empty records.")

    return {
        "epoch": epoch + 1,
        step_key: global_step,
        "mode": mode,
        "avg_total_loss": sum(x["total_loss"] for x in records) / len(records),
        "avg_ce_loss": sum(x["ce_loss"] for x in records) / len(records),
        "avg_kd_loss": sum(x["kd_loss"] for x in records) / len(records),
        "avg_hidden_loss": sum(x["hidden_loss"] for x in records) / len(records),
        "avg_attn_loss": sum(x["attn_loss"] for x in records) / len(records),
        "avg_tokens_per_sec": sum(x["tokens_per_sec"] for x in records) / len(records),
        "avg_step_time_sec": sum(x["step_time_sec"] for x in records) / len(records),
    }


class DistillationTrainer:
    def __init__(self, teacher, student, vocab_projector, tokenizer, config, logger):
        self.teacher = teacher
        self.student = student
        self.vocab_projector = vocab_projector
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger

        self.device = get_device(config)
        self.amp_dtype = get_amp_dtype(config)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_metrics_path = self.output_dir / "train_metrics.jsonl"
        self.epoch_metrics_path = self.output_dir / "epoch_metrics.jsonl"

        self.global_step = 0
        self.best_loss = float("inf")

        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        with (self.output_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

    def trainable_parameters(self) -> List[nn.Parameter]:
        params = list(self.student.parameters()) + list(self.vocab_projector.parameters())
        return params

    def build_optimizer(self) -> AdamW:
        return AdamW(
            self.trainable_parameters(),
            lr=self.config.learning_rate,
            fused=torch.cuda.is_available(),
        )

    def build_scheduler(self) -> CosineAnnealingLR:
        steps_per_epoch = math.ceil(
            self.config.data_size / self.config.per_device_train_batch_size
        )
        total_steps = max(
            1,
            steps_per_epoch
            * self.config.num_train_epochs
            // self.config.gradient_accumulation_steps,
        )

        return CosineAnnealingLR(self.optimizer, T_max=total_steps)

    @property
    def ignore_index(self) -> int:
        return self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100

    def compute_ce_loss(
        self,
        projected_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shifted_logits = projected_logits[..., :-1, :].contiguous().float()
        shifted_labels = labels[..., 1:].contiguous()

        return F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=self.ignore_index,
        )

    #Compute the KL divergence loss between the student and teacher logits
    def compute_kd_loss(
        self,
        projected_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shifted_student_logits = projected_logits[..., :-1, :].contiguous().float()
        shifted_teacher_logits = teacher_logits[..., :-1, :].contiguous().float()
        shifted_labels = labels[..., 1:].contiguous()

        student_logits = shifted_student_logits.view(-1, shifted_student_logits.size(-1))
        teacher_logits = shifted_teacher_logits.view(-1, shifted_teacher_logits.size(-1))
        labels_flat = shifted_labels.view(-1)

        temperature = self.config.temperature
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        token_kd_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)

        valid_tokens = labels_flat != self.ignore_index

        if not valid_tokens.any():
            return torch.tensor(0.0, device=self.device)

        return (temperature ** 2) * token_kd_loss[valid_tokens].mean()

    #Compute the loss for a batch of data
    def forward_teacher(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=torch.cuda.is_available(),
            ):
                return self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                    use_cache=False,
                )
    
    #calculate the loss for the student model
    def forward_student(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        with torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
            enabled=torch.cuda.is_available(),
        ):
            student_outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
            student_hidden = student_outputs.hidden_states[-1]
            projected_logits = self.vocab_projector(student_hidden)

        return student_outputs, student_hidden, projected_logits

    def compute_batch_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        teacher_outputs = self.forward_teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        _, student_hidden, projected_logits = self.forward_student(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        ce_loss = torch.tensor(0.0, device=self.device)
        kd_loss = self.compute_kd_loss(projected_logits, teacher_outputs.logits, labels)
        hidden_loss = torch.tensor(0.0, device=self.device)
        attn_loss = torch.tensor(0.0, device=self.device)
        total_loss = kd_loss

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "kd_loss": kd_loss,
            "hidden_loss": hidden_loss,
            "attn_loss": attn_loss,
        }

    @staticmethod
    def to_float(value: torch.Tensor) -> float:
        return float(value.detach().cpu().item())

    def make_step_record(
        self,
        losses: Dict[str, torch.Tensor],
        step_time: float,
        batch_tokens: int,
    ) -> Dict[str, float]:
        return {
            "total_loss": self.to_float(losses["total_loss"]),
            "ce_loss": self.to_float(losses["ce_loss"]),
            "kd_loss": self.to_float(losses["kd_loss"]),
            "hidden_loss": self.to_float(losses["hidden_loss"]),
            "attn_loss": self.to_float(losses["attn_loss"]),
            "step_time_sec": float(step_time),
            "tokens_per_sec": float(batch_tokens / max(step_time, 1e-8)),
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        step_start = time.time()

        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        losses = self.compute_batch_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss_for_backward = losses["total_loss"] / self.config.gradient_accumulation_steps
        loss_for_backward.backward()

        step_time = time.time() - step_start
        batch_tokens = int(attention_mask.sum().item())

        return self.make_step_record(losses, step_time, batch_tokens)

    def optimizer_step(self) -> None:
        torch.nn.utils.clip_grad_norm_(
            self.trainable_parameters(),
            self.config.grad_clip,
        )

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def log_if_needed(self, epoch: int, records: List[Dict[str, float]]) -> None:
        if self.global_step % self.config.logging_steps != 0:
            return

        recent = records[-self.config.logging_steps:]
        metrics = average_records(
            records=recent,
            epoch=epoch,
            global_step=self.global_step,
            mode=self.config.mode,
        )
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])

        append_jsonl(self.train_metrics_path, metrics)

        self.logger.info(
            (
                "Step %s | Loss: %.4f | CE: %.4f | KD: %.4f | "
                "Hidden: %.4f | Attn: %.4f | Tok/s: %.2f"
            ),
            self.global_step,
            metrics["avg_total_loss"],
            metrics["avg_ce_loss"],
            metrics["avg_kd_loss"],
            metrics["avg_hidden_loss"],
            metrics["avg_attn_loss"],
            metrics["avg_tokens_per_sec"],
        )

    def save_if_needed(self, epoch: int, records: List[Dict[str, float]]) -> None:
        if self.global_step % self.config.save_steps != 0:
            return

        recent = records[-max(1, self.config.logging_steps):]
        metrics = average_records(
            records=recent,
            epoch=epoch,
            global_step=self.global_step,
            mode=self.config.mode,
        )

        self.save_checkpoint(step=self.global_step, metrics=metrics)

        if metrics["avg_total_loss"] < self.best_loss:
            self.best_loss = metrics["avg_total_loss"]
            self.save_checkpoint(step=self.global_step, metrics=metrics, tag="best")

    def cleanup_old_checkpoints(self, keep_last: int = 2) -> None:
        checkpoint_pattern = re.compile(r"checkpoint-(\d+)$")
        checkpoints = []

        for path in self.output_dir.iterdir():
            match = checkpoint_pattern.match(path.name)

            if path.is_dir() and match:
                checkpoints.append((int(match.group(1)), path))

        checkpoints.sort(key=lambda item: item[0])

        for _, path in checkpoints[:-keep_last]:
            shutil.rmtree(path, ignore_errors=True)
            self.logger.info("Deleted old checkpoint: %s", path)

    def save_checkpoint(
        self,
        step: int,
        metrics: Dict[str, Any],
        tag: Optional[str] = None,
    ) -> None:
        checkpoint_name = tag or f"checkpoint-{step}"
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.student.save_pretrained(checkpoint_dir / "student")
        self.tokenizer.save_pretrained(checkpoint_dir / "student")

        torch.save(
            self.vocab_projector.state_dict(),
            checkpoint_dir / "vocab_projector.pt",
        )

        with (checkpoint_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        if tag is None:
            self.cleanup_old_checkpoints(keep_last=2)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, Any]:
        self.student.train()
        self.vocab_projector.train()

        epoch_records: List[Dict[str, float]] = []
        self.optimizer.zero_grad(set_to_none=True)

        for local_step, batch in enumerate(dataloader, start=1):
            record = self.train_step(batch)
            epoch_records.append(record)

            if local_step % self.config.gradient_accumulation_steps == 0:
                self.optimizer_step()
                self.log_if_needed(epoch, epoch_records)
                self.save_if_needed(epoch, epoch_records)

        epoch_summary = average_records(
            records=epoch_records,
            epoch=epoch,
            global_step=self.global_step,
            mode=self.config.mode,
            step_key="global_step_end",
        )

        append_jsonl(self.epoch_metrics_path, epoch_summary)

        self.logger.info(
            (
                "Epoch %s finished | Total: %.4f | CE: %.4f | KD: %.4f | "
                "Hidden: %.4f | Attn: %.4f | Tok/s: %.2f"
            ),
            epoch + 1,
            epoch_summary["avg_total_loss"],
            epoch_summary["avg_ce_loss"],
            epoch_summary["avg_kd_loss"],
            epoch_summary["avg_hidden_loss"],
            epoch_summary["avg_attn_loss"],
            epoch_summary["avg_tokens_per_sec"],
        )

        return epoch_summary

    def train(self, dataloader: DataLoader) -> None:
        epoch_summaries = []

        for epoch in range(self.config.num_train_epochs):
            epoch_summaries.append(self.train_epoch(dataloader, epoch))

        final_metrics = epoch_summaries[-1] if epoch_summaries else {"mode": self.config.mode}
        self.save_checkpoint(self.global_step, final_metrics, tag="final")


def load_tokenizer(config: DistillationConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.teacher_model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_teacher(config: DistillationConfig, amp_dtype: torch.dtype):
    device_index = get_device_index(config.device)

    if config.load_teacher_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=amp_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            quantization_config=quantization_config,
            device_map={"": device_index},
            attn_implementation=config.attn_implementation,
            trust_remote_code=True,
        )
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            torch_dtype=amp_dtype,
            device_map={"": device_index},
            attn_implementation=config.attn_implementation,
            trust_remote_code=True,
        )

    freeze_model(teacher)
    return teacher


def load_student(
    config: DistillationConfig,
    amp_dtype: torch.dtype,
    device: torch.device,
    logger: logging.Logger,
):
    if config.resume_from_checkpoint is not None:
        student_path = Path(config.resume_from_checkpoint) / "student"
        logger.info("Resuming student from checkpoint: %s", student_path)
    else:
        student_path = config.student_model

    student = AutoModelForCausalLM.from_pretrained(
        student_path,
        torch_dtype=amp_dtype,
        attn_implementation=config.attn_implementation,
        trust_remote_code=True,
    ).to(device)

    if config.use_gradient_checkpointing:
        student.gradient_checkpointing_enable()

        if hasattr(student.config, "use_cache"):
            student.config.use_cache = False

    return student


def load_models(config: DistillationConfig, logger: logging.Logger):
    device = get_device(config)
    amp_dtype = get_amp_dtype(config)

    tokenizer = load_tokenizer(config)
    teacher = load_teacher(config, amp_dtype)
    student = load_student(config, amp_dtype, device, logger)

    if student.config.vocab_size < teacher.config.vocab_size:
        logger.info(
            "Resizing student embeddings: %s -> %s",
            student.config.vocab_size,
            teacher.config.vocab_size,
        )
        student.resize_token_embeddings(teacher.config.vocab_size)

    vocab_projector = VocabProjector(
        student.config.hidden_size,
        teacher.config.vocab_size,
    ).to(device=device, dtype=amp_dtype)

    if config.resume_from_checkpoint is not None:
        checkpoint_dir = Path(config.resume_from_checkpoint)

        vocab_projector_path = checkpoint_dir / "vocab_projector.pt"
        if vocab_projector_path.exists():
            logger.info("Loading vocab projector from: %s", vocab_projector_path)
            vocab_projector.load_state_dict(
                torch.load(vocab_projector_path, map_location="cpu")
            )

    logger.info("Teacher hidden size: %s", teacher.config.hidden_size)
    logger.info("Student hidden size: %s", student.config.hidden_size)
    logger.info("Teacher vocab size: %s", teacher.config.vocab_size)
    logger.info("Student vocab size: %s", student.config.vocab_size)

    logger.info("Trainable student params: %s", f"{count_trainable_parameters(student):,}")
    logger.info(
        "Trainable vocab projector params: %s",
        f"{count_trainable_parameters(vocab_projector):,}",
    )
    return teacher, student, vocab_projector, tokenizer


def build_dataloader(config: DistillationConfig, tokenizer) -> DataLoader:
    dataset = DistillationDataset(
        tokenizer=tokenizer,
        data_path=config.data_path,
        data_size=config.data_size,
        max_seq_length=config.max_seq_length,
    )

    return DataLoader(
        dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def run_training(config: DistillationConfig) -> None:
    logger = setup_logger(config.output_dir)

    logger.info("Starting training")
    logger.info(json.dumps(asdict(config), indent=2, ensure_ascii=False))

    teacher, student, vocab_projector, tokenizer = load_models(config, logger)
    dataloader = build_dataloader(config, tokenizer)

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        vocab_projector=vocab_projector,
        tokenizer=tokenizer,
        config=config,
        logger=logger,
    )

    trainer.train(dataloader)


if __name__ == "__main__":
    config = DistillationConfig()
    run_training(config)
