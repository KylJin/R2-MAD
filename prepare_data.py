import os
import json
import argparse
import random

from src.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


DATASET_NAMES = ["math500", "mmlu_pro_engineering", "mmlu_pro_economics", "truthfulqa"]


def load_data_from_json(file_path:str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dump_data_to_json(data:list, file_path:str):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")


def math500_processor():
    if not os.path.exists(os.path.join(PROCESSED_DATA_DIR, "math500")):
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, "math500"))
    
    source_path = os.path.join(RAW_DATA_DIR, "MATH500", "test.json")

    train_data, test_data = [], []
    with open(source_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            sample = {
                "question": entry["problem"],
                "answer": entry["answer"],
                "solution": entry["solution"],
                "category": entry["subject"],
            }
            if entry.get("level") == 5:
                test_data.append(sample)
            else:
                train_data.append(sample)

    dump_data_to_json(train_data, os.path.join(PROCESSED_DATA_DIR, "math500", "train.jsonl"))
    dump_data_to_json(test_data, os.path.join(PROCESSED_DATA_DIR, "math500", "test.jsonl"))


def _mmlu_pro_subset_processor(category: str, seed: int = 42):
    dataset_path = os.path.join(RAW_DATA_DIR, "MMLU-PRO", "mmlu_pro_test.json")
    subset_data = []
    with open(dataset_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("category", "").lower() != category.lower():
                continue
            options = entry["options"]
            option_keys = [chr(ord("A") + i) for i in range(len(options))]
            sample = {
                "question": entry["question"],
                "options": dict(zip(option_keys, options)),
                "answer": entry["answer"],
                "category": entry["category"],
            }
            subset_data.append(sample)
    
    random.seed(seed)
    random.shuffle(subset_data)
    split = int(len(subset_data) * 0.75)
    train_data, test_data = subset_data[:split], subset_data[split:]

    out_dir = os.path.join(PROCESSED_DATA_DIR, "mmlu_pro", category.lower())
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dump_data_to_json(train_data, os.path.join(out_dir, "train.jsonl"))
    dump_data_to_json(test_data, os.path.join(out_dir, "test.jsonl"))


def mmlu_pro_engineering_processor(seed: int = 42):
    _mmlu_pro_subset_processor("engineering", seed=seed)


def mmlu_pro_economics_processor(seed: int = 42):
    _mmlu_pro_subset_processor("economics", seed=seed)


def truthfulqa_processor(seed: int = 42):
    if not os.path.exists(os.path.join(PROCESSED_DATA_DIR, "truthfulqa")):
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, "truthfulqa"))

    source_path = os.path.join(RAW_DATA_DIR, "TruthfulQA", "multiple_choice_validation.json")

    random.seed(seed)
    data = []
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            choices = entry["mc1_targets"]["choices"]
            labels = entry["mc1_targets"]["labels"]
            if not (4 <= len(choices) <= 9):
                continue
            correct_choice = choices[labels.index(1)]
            shuffled_choices = choices[:]
            random.shuffle(shuffled_choices)
            option_keys = [chr(ord("A") + i) for i in range(len(shuffled_choices))]
            answer = option_keys[shuffled_choices.index(correct_choice)]
            sample = {
                "question": entry["question"],
                "options": dict(zip(option_keys, shuffled_choices)),
                "answer": answer,
                "category": "truthfulqa",
            }
            data.append(sample)

    random.shuffle(data)
    split = int(len(data) * 0.75)
    train_data, test_data = data[:split], data[split:]

    dump_data_to_json(train_data, os.path.join(PROCESSED_DATA_DIR, "truthfulqa", "train.jsonl"))
    dump_data_to_json(test_data, os.path.join(PROCESSED_DATA_DIR, "truthfulqa", "test.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_NAMES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset_name == "math500":
        math500_processor()
        print(f"Processed {args.dataset_name} dataset")

    elif args.dataset_name == "mmlu_pro_engineering":
        mmlu_pro_engineering_processor(seed=args.seed)
        print(f"Processed {args.dataset_name} dataset")

    elif args.dataset_name == "mmlu_pro_economics":
        mmlu_pro_economics_processor(seed=args.seed)
        print(f"Processed {args.dataset_name} dataset")

    elif args.dataset_name == "truthfulqa":
        truthfulqa_processor(seed=args.seed)
        print(f"Processed {args.dataset_name} dataset")
