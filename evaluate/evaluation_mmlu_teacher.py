import os
import sys
import argparse
import re
import torch
import pandas as pd
from tqdm import tqdm
from thefuzz import process
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


'''
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../

pip install thefuzz
python evaluation_mmlu_teacher.py -d data/mmlu/data/
'''


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval()

    try:
        model.generation_config = GenerationConfig.from_pretrained(
            args.checkpoint_path,
            trust_remote_code=True
        )
    except Exception:
        pass

    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.0

    # Avoid useless warning when do_sample=False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return model, tokenizer


def format_example(line):
    example = (
        "The following is a multiple-choice question. "
        "Please choose the most suitable one among A, B, C and D as the answer to this question.\n\n"
        + line["question"]
        + "\n"
    )

    for choice in choices:
        example += f'{choice}. {line[f"{choice}"]}\n'

    example += "\nYour answer is:"
    return example


def process_before_extraction(gen, choice_dict):
    choice_dict = {key: str(val) for key, val in choice_dict.items()}

    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
        gen = pattern.sub(key, gen)

    return gen


def extract_choice(gen, choice_list):
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
            return choices[choice_list.index(best_match)]
        except Exception:
            return "A"

    return res.group(1)


def extract_answer(response, row):
    gen = process_before_extraction(
        response,
        {choice: row[choice] for choice in choices}
    )
    pred = extract_choice(gen, [row[choice] for choice in choices])
    return pred


def generate_response(model, tokenizer, question):
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
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_length = inputs["input_ids"].shape[-1]

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()

    return response


@torch.no_grad()
def eval_subject(
    model,
    tokenizer,
    subject_name,
    test_df,
    save_result_dir=None,
    overwrite=False,
    debug=False,
    **kwargs
):
    os.makedirs(save_result_dir, exist_ok=True)

    result_path = os.path.join(save_result_dir, f"{subject_name}_result.csv")

    if not overwrite and os.path.exists(result_path):
        print(f"{result_path} existed, skip!", flush=True)

        score = []
        df_result = pd.read_csv(result_path).astype(str)

        for (_, datarow), (_, resultrow) in zip(test_df.iterrows(), df_result.iterrows()):
            pred = resultrow["model_output"]
            correct = 1 if pred == datarow["answer"] else 0
            score.append(correct)

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
        file=sys.stdout
    ):
        question = format_example(row)

        response = generate_response(model, tokenizer, question)
        pred = extract_answer(response, row)

        if "answer" in row:
            correct = 1 if pred == row["answer"] else 0
            score.append(correct)

        result.append(pred)
        responses.append(response)

        if debug:
            print(question, flush=True)
            print(response, flush=True)
            print(pred, flush=True)
            if "answer" in row:
                print(f'pred: {pred} ref: {row["answer"]} correct: {correct}', flush=True)
            print("======================", flush=True)

    if save_result_dir:
        test_df = test_df.copy()
        test_df["model_output"] = result
        test_df["model_response"] = responses

        if score:
            test_df["correctness"] = score

        test_df.to_csv(
            result_path,
            encoding="utf-8",
            index=False,
        )

    return score


def cal_mmlu(res):
    acc_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0

    for class_ in TASK_NAME_MAPPING.keys():
        acc_sum_dict[class_] = 0.0
        cnt_dict[class_] = 0.0

        for tt in TASK_NAME_MAPPING[class_]:
            if tt not in res:
                continue

            acc_sum += sum(res[tt])
            cnt += len(res[tt])

            acc_sum_dict[class_] += sum(res[tt])
            cnt_dict[class_] += len(res[tt])

    print("\n\n\n", flush=True)

    for k in TASK_NAME_MAPPING.keys():
        if k in cnt_dict and cnt_dict[k] > 0:
            print("%s ACC: %.2f" % (k, acc_sum_dict[k] * 100 / cnt_dict[k]), flush=True)

    if cnt > 0:
        print("AVERAGE ACC: %.2f" % (acc_sum * 100 / cnt), flush=True)
    else:
        print("No valid examples evaluated.", flush=True)


def main(args):
    print("loading model weights", flush=True)

    if args.checkpoint_path is not None:
        model, tokenizer = load_models_tokenizer(args)
    else:
        model, tokenizer = None, None

    print("model loaded", flush=True)
    print(f"checkpoint path: {args.checkpoint_path}", flush=True)
    print(f"eval data path: {args.eval_data_path}", flush=True)

    save_result_dir = os.path.join(
        "outs_chat",
        "mmlu_eval_result_teacher",
        args.checkpoint_path.replace("/", "_")
    )

    print(f"save result dir: {save_result_dir}", flush=True)

    dev_result = {}

    for subject_name in tqdm(
        SUBJECTS,
        desc="MMLU subjects",
        file=sys.stdout
    ):
        print(f"\nStarting subject: {subject_name}", flush=True)

        test_file_path = os.path.join(
            args.eval_data_path,
            "test",
            f"{subject_name}_test.csv"
        )

        if not os.path.exists(test_file_path):
            print(f"Skip {subject_name}: file not found at {test_file_path}", flush=True)
            continue

        test_df = pd.read_csv(
            test_file_path,
            names=["question", "A", "B", "C", "D", "answer"],
            keep_default_na=False
        ).astype(str)

        score = eval_subject(
            model=model,
            tokenizer=tokenizer,
            subject_name=subject_name,
            test_df=test_df,
            save_result_dir=save_result_dir,
            overwrite=args.overwrite,
            debug=args.debug,
        )

        dev_result[subject_name] = score

        if len(score) > 0:
            subject_acc = sum(score) * 100 / len(score)
            print(
                f"Finished {subject_name}: ACC = {subject_acc:.2f}%, N = {len(score)}",
                flush=True
            )
        else:
            print(f"Finished {subject_name}: no valid examples.", flush=True)

    cal_mmlu(dev_result)


TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}

SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
choices = ["A", "B", "C", "D"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")

    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="Qwen/Qwen2.5-3B-Instruct",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1234,
        help="Random seed"
    )

    group = parser.add_argument_group(title="Evaluation options")

    group.add_argument(
        "-d",
        "--eval_data_path",
        type=str,
        default="data/mmlu/data",
        help="Path to eval data"
    )

    group.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Only evaluate first 5 examples per subject and print details."
    )

    group.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing results."
    )

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)