"This file is used to download necessary raw datasets for the project."
import os
import argparse
import shutil
from datasets import load_dataset

from src.paths import RAW_DATA_DIR


DATASET_NAMES = ["math500", "mmlu_pro", "truthfulqa"]


def math500(target_dir:str):
    """Download MATH500 dataset to the target directory.

    Args:
        target_dir: str, the directory to save the dataset.

    Returns:
        None
    """
    dataset_id = "HuggingFaceH4/MATH-500"
    target_dataset_dir = os.path.join(target_dir, "MATH500")
    if not os.path.exists(target_dataset_dir):
        os.makedirs(target_dataset_dir)

    dataset = load_dataset("HuggingFaceH4/MATH-500", "default", cache_dir=target_dir)
    test_set = dataset["test"]
    print(f"DATASET INFO: {len(test_set)} test samples")
    test_set.to_json(os.path.join(target_dataset_dir, "test.json"))

    tmp_dir_name = f"{dataset_id.split('/')[0]}___{dataset_id.split('/')[1].lower()}"
    shutil.rmtree(os.path.join(target_dir, tmp_dir_name))

    lockfiles = [file for file in os.listdir(target_dir) if file.endswith(".lock")]
    for lockfile in lockfiles:
        os.remove(os.path.join(target_dir, lockfile))

    print(f"MATH500 dataset downloaded to {target_dataset_dir}")


def mmlu_pro(target_dir:str):
    """Download MMLU-Pro dataset to the target directory.
    
    Args:
        target_dir: str, the directory to save the dataset.
        
    Returns:
        None
    """
    dataset_id = "TIGER-Lab/MMLU-Pro"
    target_dataset_dir = os.path.join(target_dir, "MMLU-PRO")
    if not os.path.exists(target_dataset_dir):
        os.makedirs(target_dataset_dir)
    
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", "default", cache_dir=target_dir)
    test_set = dataset["test"]
    print(f"DATASET INFO: {len(test_set)} test samples")
    test_set.to_json(os.path.join(target_dataset_dir, "mmlu_pro_test.json"))
    
    tmp_dir_name = f"{dataset_id.split('/')[0]}___{dataset_id.split('/')[1].lower()}"
    shutil.rmtree(os.path.join(target_dir, tmp_dir_name))
    
    lockfiles = [file for file in os.listdir(target_dir) if file.endswith(".lock")]
    for lockfile in lockfiles:
        os.remove(os.path.join(target_dir, lockfile))
    
    print(f"MMLU-Pro dataset downloaded to {target_dataset_dir}")


def truthfulqa(target_dir:str):
    """Download TruthfulQA dataset to the target directory.

    Args:
        target_dir: str, the directory to save the dataset.

    Returns:
        None
    """
    dataset_id = "truthfulqa/truthful_qa"
    target_dataset_dir = os.path.join(target_dir, "TruthfulQA")
    if not os.path.exists(target_dataset_dir):
        os.makedirs(target_dataset_dir)

    for config in ["generation", "multiple_choice"]:
        dataset = load_dataset(dataset_id, config, cache_dir=target_dir)
        validation_set = dataset["validation"]
        print(f"DATASET ({config}) INFO: {len(validation_set)} validation samples")
        validation_set.to_json(os.path.join(target_dataset_dir, f"{config}_validation.json"))

    tmp_dir_name = f"{dataset_id.split('/')[0]}___{dataset_id.split('/')[1].lower()}"
    shutil.rmtree(os.path.join(target_dir, tmp_dir_name))

    lockfiles = [file for file in os.listdir(target_dir) if file.endswith(".lock")]
    for lockfile in lockfiles:
        os.remove(os.path.join(target_dir, lockfile))

    print(f"TruthfulQA dataset downloaded to {target_dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_NAMES)
    parser.add_argument("--dataset_dir", type=str, default=RAW_DATA_DIR)
    args = parser.parse_args()

    dataset_target_dir = args.dataset_dir
    
    if not os.path.exists(dataset_target_dir):
        os.makedirs(dataset_target_dir)
    
    if args.dataset_name == "math500":
        if not os.path.exists(os.path.join(dataset_target_dir, "MATH500")):
            print(f"Downloading MATH500 dataset to {dataset_target_dir}...")
            math500(dataset_target_dir)
        else:
            print(f"MATH500 dataset already exists in {dataset_target_dir}")
    
    elif args.dataset_name == "mmlu_pro":
        if not os.path.exists(os.path.join(dataset_target_dir, "MMLU-PRO")):
            print(f"Downloading MMLU-Pro dataset to {dataset_target_dir}...")
            mmlu_pro(dataset_target_dir)
        else:
            print(f"MMLU-Pro dataset already exists in {dataset_target_dir}")

    elif args.dataset_name == "truthfulqa":
        if not os.path.exists(os.path.join(dataset_target_dir, "TruthfulQA")):
            print(f"Downloading TruthfulQA dataset to {dataset_target_dir}...")
            truthfulqa(dataset_target_dir)
        else:
            print(f"TruthfulQA dataset already exists in {dataset_target_dir}")
