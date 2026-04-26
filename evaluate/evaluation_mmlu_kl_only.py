import os
import sys
import json
import argparse
import re
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from thefuzz import process
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPERIMENT_NAME = "kl_only"
DEFAULT_CHECKPOINT = "/nfs/speed-scratch/z_yuefan/distill-project/original/runs/qwen3b_smol360m_kl_only/best"

TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics",
        "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios",
        "philosophy", "prehistory", "professional_law", "world_religions",
    ],
    "other": [
        "business_ethics", "college_medicine", "human_aging", "management",
        "marketing", "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
        "global_facts", "clinical_knowledge",
    ],
    "social": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
}

SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
CHOICES = ["A", "B", "C", "D"]


class VocabProjector(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projector = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.projector(x)


def resolve_checkpoint_paths(checkpoint_path: str):
    """
    Training saved:
      best/
        metrics.json
        vocab_projector.pt
        student/
          config.json
          model weights
          tokenizer files

    Therefore:
      model/tokenizer path = best/student
      projector path       = best/vocab_projector.pt
    """
    ckpt = Path(checkpoint_path)

    if (ckpt / "student").is_dir():
        ckpt_root = ckpt
        student_dir = ckpt / "student"
    elif ckpt.name == "student":
        student_dir = ckpt
        ckpt_root = ckpt.parent
    else:
        ckpt_root = ckpt
        student_dir = ckpt

    projector_path = ckpt_root / "vocab_projector.pt"
    return ckpt_root, student_dir, projector_path


def load_model_tokenizer_projector(args):
    ckpt_root, student_dir, projector_path = resolve_checkpoint_paths(args.checkpoint_path)

    print(f"checkpoint root: {ckpt_root}", flush=True)
    print(f"student dir: {student_dir}", flush=True)
    print(f"projector path: {projector_path}", flush=True)

    print("loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        student_dir,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading student model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        student_dir,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).eval()

    projector = None

    if args.use_projector:
        if not projector_path.exists():
            raise FileNotFoundError(
                f"Expected vocab_projector.pt at {projector_path}, but it does not exist. "
                "Use --no_projector if you intentionally want native lm_head evaluation."
            )

        print("loading vocab_projector.pt", flush=True)
        state = torch.load(projector_path, map_location="cpu")
        weight = state["projector.weight"]
        vocab_size, d_model = weight.shape

        projector = VocabProjector(d_model=d_model, vocab_size=vocab_size)
        projector.load_state_dict(state)
        projector = projector.to(device=model.device, dtype=torch.float16).eval()

        print(f"projector loaded: d_model={d_model}, vocab_size={vocab_size}", flush=True)
    else:
        print("WARNING: using native lm_head instead of vocab_projector.", flush=True)

    return model, tokenizer, projector


def format_example(line):
    example = (
        "The following is a multiple-choice question. "
        "Please choose the most suitable one among A, B, C and D as the answer to this question.\n\n"
        + line["question"]
        + "\n"
    )
    for choice in CHOICES:
        example += f'{choice}. {line[f"{choice}"]}\n'
    example += "\nYour answer is:"
    return example


def encode_prompt(tokenizer, question, device):
    messages = [{"role": "user", "content": question}]
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except Exception:
        prompt = f"User: {question}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def generate_response(model, tokenizer, projector, question, max_new_tokens=16):
    inputs = encode_prompt(tokenizer, question, model.device)

    if projector is None:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_length = inputs["input_ids"].shape[-1]
        return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    generated = []

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        last_hidden = outputs.hidden_states[-1][:, -1, :]
        logits = projector(last_hidden)

        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = int(next_token.item())

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, device=attention_mask.device)],
            dim=-1,
        )

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def process_before_extraction(gen, choice_dict):
    choice_dict = {key: str(val) for key, val in choice_dict.items()}
    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
        gen = pattern.sub(key, gen)
    return gen


def extract_choice(gen, choice_list):
    gen = str(gen).strip()

    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        try:
            choice_list = [str(x) for x in choice_list]
            best_match = process.extractOne(gen, choice_list)[0]
            return CHOICES[choice_list.index(best_match)]
        except Exception:
            return "A"

    return res.group(1)


def extract_answer(response, row):
    gen = process_before_extraction(
        response,
        {choice: row[choice] for choice in CHOICES},
    )
    return extract_choice(gen, [row[choice] for choice in CHOICES])


@torch.no_grad()
def eval_subject(
    model,
    tokenizer,
    projector,
    subject_name,
    test_df,
    save_result_dir,
    overwrite=False,
    debug=False,
    max_new_tokens=16,
):
    os.makedirs(save_result_dir, exist_ok=True)
    result_path = os.path.join(save_result_dir, f"{subject_name}_result.csv")

    if not overwrite and os.path.exists(result_path):
        print(f"{result_path} existed, skip!", flush=True)
        score = []
        df_result = pd.read_csv(result_path).astype(str)
        for (_, datarow), (_, resultrow) in zip(test_df.iterrows(), df_result.iterrows()):
            pred = resultrow["model_output"]
            score.append(1 if pred == datarow["answer"] else 0)
        return score

    result = []
    responses = []
    score = []

    if debug:
        test_df = test_df.iloc[:5]

    for _, row in tqdm(
        test_df.iterrows(),
        total=len(test_df),
        desc=subject_name,
        file=sys.stdout,
    ):
        question = format_example(row)
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            projector=projector,
            question=question,
            max_new_tokens=max_new_tokens,
        )

        pred = extract_answer(response, row)
        correct = 1 if pred == row["answer"] else 0

        score.append(correct)
        result.append(pred)
        responses.append(response)

        if debug:
            print(question, flush=True)
            print(response, flush=True)
            print(f'pred: {pred} ref: {row["answer"]} correct: {correct}', flush=True)
            print("======================", flush=True)

    test_df = test_df.copy()
    test_df["model_output"] = result
    test_df["model_response"] = responses
    test_df["correctness"] = score
    test_df.to_csv(result_path, encoding="utf-8", index=False)

    return score


def cal_mmlu(res):
    acc_sum_dict = {}
    cnt_dict = {}
    acc_sum = 0.0
    cnt = 0

    for class_, subjects in TASK_NAME_MAPPING.items():
        acc_sum_dict[class_] = 0.0
        cnt_dict[class_] = 0.0
        for subject in subjects:
            if subject not in res:
                continue
            acc_sum += sum(res[subject])
            cnt += len(res[subject])
            acc_sum_dict[class_] += sum(res[subject])
            cnt_dict[class_] += len(res[subject])

    summary = {
        "experiment": EXPERIMENT_NAME,
        "category_accuracy": {},
        "average_accuracy": None,
        "num_examples": int(cnt),
    }

    print("\n\n\n", flush=True)

    for k in TASK_NAME_MAPPING.keys():
        if cnt_dict.get(k, 0) > 0:
            acc = acc_sum_dict[k] * 100 / cnt_dict[k]
            summary["category_accuracy"][k] = acc
            print("%s ACC: %.2f" % (k, acc), flush=True)

    if cnt > 0:
        avg = acc_sum * 100 / cnt
        summary["average_accuracy"] = avg
        print("AVERAGE ACC: %.2f" % avg, flush=True)
    else:
        print("No valid examples evaluated.", flush=True)

    return summary


def main(args):
    print("loading model weights", flush=True)
    print(f"experiment: {EXPERIMENT_NAME}", flush=True)
    print(f"checkpoint path: {args.checkpoint_path}", flush=True)
    print(f"eval data path: {args.eval_data_path}", flush=True)
    print(f"use_projector: {args.use_projector}", flush=True)

    model, tokenizer, projector = load_model_tokenizer_projector(args)

    print("model loaded", flush=True)

    save_result_dir = os.path.join(
        "outs_chat",
        "mmlu_eval_result_distilled_projector" if args.use_projector else "mmlu_eval_result_distilled_native",
        EXPERIMENT_NAME,
        Path(args.checkpoint_path).name.replace("/", "_"),
    )
    print(f"save result dir: {save_result_dir}", flush=True)

    dev_result = {}

    for subject_name in tqdm(SUBJECTS, desc="MMLU subjects", file=sys.stdout):
        print(f"\nStarting subject: {subject_name}", flush=True)

        test_file_path = os.path.join(
            args.eval_data_path,
            "test",
            f"{subject_name}_test.csv",
        )

        if not os.path.exists(test_file_path):
            print(f"Skip {subject_name}: file not found at {test_file_path}", flush=True)
            continue

        test_df = pd.read_csv(
            test_file_path,
            names=["question", "A", "B", "C", "D", "answer"],
            keep_default_na=False,
        ).astype(str)

        score = eval_subject(
            model=model,
            tokenizer=tokenizer,
            projector=projector,
            subject_name=subject_name,
            test_df=test_df,
            save_result_dir=save_result_dir,
            overwrite=args.overwrite,
            debug=args.debug,
            max_new_tokens=args.max_new_tokens,
        )

        dev_result[subject_name] = score

        if len(score) > 0:
            subject_acc = sum(score) * 100 / len(score)
            print(f"Finished {subject_name}: ACC = {subject_acc:.2f}%, N = {len(score)}", flush=True)
        else:
            print(f"Finished {subject_name}: no valid examples.", flush=True)

    summary = cal_mmlu(dev_result)

    summary_path = os.path.join(save_result_dir, "summary.json")
    Path(summary_path).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"summary saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate distilled model on MMLU: {EXPERIMENT_NAME}")

    parser.add_argument(
        "-c",
        "--checkpoint-path",
        "--model_name",
        dest="checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint root, e.g. runs/.../best. The script will load best/student and best/vocab_projector.pt.",
    )
    parser.add_argument(
        "-d",
        "--eval_data_path",
        type=str,
        default="/nfs/speed-scratch/z_yuefan/distill-project/original/data/mmlu/data",
        help="Path to extracted MMLU data directory.",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--no_projector",
        dest="use_projector",
        action="store_false",
        help="Use native model lm_head instead of vocab_projector.pt. Not recommended for your current training code.",
    )
    parser.set_defaults(use_projector=True)

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
