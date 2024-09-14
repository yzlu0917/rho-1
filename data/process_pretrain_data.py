import os
import random

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def process_split(split_name, tokenizer, data_path, output_path):
    """
    处理数据集的一个分割部分（训练、测试或验证）并保存为二进制文件。

    Args:
    - split_name: 数据集的分割名称，如 'train', 'test', 'validation'。
    - tokenizer: 分词器对象。
    - data_path: 数据集的路径。
    - output_path: 处理后数据保存的路径。
    """
    
    dataset = load_dataset(data_path, split=split_name, streaming=True)
    with open(output_path, 'wb') as f:
        for line in tqdm(dataset, desc=f"Processing {split_name}", mininterval=1.0):
            if line['meta']['redpajama_set_name'] in ['RedPajamaC4', 'RedPajamaCommonCrawl'] and random.random() > 0.15:
                continue
            else:
                text = line['text']
                try:
                    text_id = tokenizer.encode(text, add_special_tokens=False)
                    text_id.append(tokenizer.eos_token_id)
                    arr = np.array(text_id, dtype=np.uint16)
                    f.write(arr.tobytes())
                except Exception as e:
                    print(f"Error processing line: {line}\nException: {e}")

    print(f"{split_name.capitalize()} data saved to {output_path}")


def process_slim_pajama(data_path="/cephfs/shared/hf_cache/datasets/SlimPajama-627B",
                        output_dir="/cephfs/shared/lichao/pretrain/data/filted_slim_pajama_627b",
                        tokenizer_name="huggyllama/llama-7b"
                        ):
    """
    处理 Slim Pajama 数据集，将其转换为二进制格式保存。

    Args:
    - data_path: 数据集路径。
    - output_dir: 处理后数据保存的目录。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    os.makedirs(output_dir, exist_ok=True)

    splits = ['test', "validation"]
    for split in splits:
        output_path = f"{output_dir}/{split}/slim_pajama_627b_{split}.bin"
        process_split(split, tokenizer, data_path, output_path)


if __name__ == "__main__":
    process_slim_pajama()
