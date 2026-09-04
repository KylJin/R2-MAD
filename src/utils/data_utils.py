import os
import json
from typing import Dict

from src.paths import PROCESSED_DATA_DIR


def read_json(file_path):
    file_path = str(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def format_choices(options: Dict[str, str]) -> str:
    return '\n'.join([idx + '. ' + item for idx, item in list(options.items())])


class DataLoader:
    @staticmethod
    def load_dataset(question_type: str) -> Dict:
        dataset_paths = {
            "MATH500": "math500/test.jsonl",
            "MATH500_TRAIN": "math500/train.jsonl",
            "MMLUPro_Engineering": "mmlu_pro/engineering/test.jsonl",
            "MMLUPro_Engineering_TRAIN": "mmlu_pro/engineering/train.jsonl",
            "MMLUPro_Economics": "mmlu_pro/economics/test.jsonl",
            "MMLUPro_Economics_TRAIN": "mmlu_pro/economics/train.jsonl",
            "TruthfulQA": "truthfulqa/test.jsonl",
            "TruthfulQA_TRAIN": "truthfulqa/train.jsonl",
        }

        if question_type not in dataset_paths:
            raise ValueError(f"DataLoader: Unsupported question type in reading data: {question_type}")

        dataset = read_json(os.path.join(PROCESSED_DATA_DIR, dataset_paths[question_type]))
        return dataset
    
    @staticmethod
    def format_question(item: Dict, question_type: str) -> tuple:
        formatters = {
            "MATH500": lambda x: (x['question'], x['answer'], x['solution'], x['category']),
            "MATH500_TRAIN": lambda x: (x['question'], x['answer'], x['solution'], x['category']),
            "MMLUPro_Engineering": lambda x: (
                f"Question: {x['question']}\n\n{format_choices(x['options'])}",
                x['answer'],
                x['answer'],
                "engineering"
            ),
            "MMLUPro_Engineering_TRAIN": lambda x: (
                f"Question: {x['question']}\n\n{format_choices(x['options'])}",
                x['answer'],
                x['answer'],
                "engineering"
            ),
            "MMLUPro_Economics": lambda x: (
                f"Question: {x['question']}\n\n{format_choices(x['options'])}",
                x['answer'],
                x['answer'],
                "economics"
            ),
            "MMLUPro_Economics_TRAIN": lambda x: (
                f"Question: {x['question']}\n\n{format_choices(x['options'])}",
                x['answer'],
                x['answer'],
                "economics"
            ),
            "TruthfulQA": lambda x: (
                f"Question: {x['question']}\n\n{format_choices(x['options'])}", 
                x['answer'], 
                x['answer'], 
                "truthfulqa"
            ),
            "TruthfulQA_TRAIN": lambda x: (
                f"Question: {x['question']}\n\n{format_choices(x['options'])}", 
                x['answer'], 
                x['answer'], 
                "truthfulqa"
            ),
        }

        if question_type not in formatters:
            raise ValueError(f"DataLoader: Unsupported question type in format question: {question_type}")

        return formatters[question_type](item)
