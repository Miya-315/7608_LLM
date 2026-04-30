
import json
import random
import os

def split_file(input_path, train_name, test_name, ratio=0.9):
    print(f"Loading {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total lines: {len(lines)}")
    random.seed(42)  # For reproducibility
    random.shuffle(lines)

    split_idx = int(len(lines) * ratio)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    dir_path = os.path.dirname(input_path)
    train_path = os.path.join(dir_path, train_name)
    test_path = os.path.join(dir_path, test_name)

    print(f"Writing {len(train_lines)} to {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    print(f"Writing {len(test_lines)} to {test_path}...")
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
    print("Done.\n")

if __name__ == "__main__":
    # 1. Split raw math dataset
    split_file(
        "/gfs/space/chatrl/users/djy/0121/sft_code/250305-250529_math_combined-41649_no_repeat_70_turn1_deepseekv3.2.jsonl",
        "raw_math_train.jsonl", # 原始数学数据集训练集
        "raw_math_test.jsonl",  # 原始数学数据集测试集
        ratio=0.9
    )

    # 2. Split labeled dataset (data.jsonl)
    split_file(
        "/gfs/space/chatrl/users/djy/0121/sft_code/data.jsonl",
        "data_train.jsonl", # 标签数据集训练集
        "data_test.jsonl",  # 标签数据集测试集
        ratio=0.9
    )
