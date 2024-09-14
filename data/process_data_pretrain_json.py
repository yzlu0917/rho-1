import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import multiprocessing
from functools import partial

def process_file(file_path, tokenizer, output_dir):
    """
    处理单个 JSON 文件并保存为二进制文件。

    Args:
    - file_path: JSON 文件的路径。
    - tokenizer: 分词器对象。
    - output_dir: 输出目录。
    """
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.bin")
    
    with open(file_path, 'r', encoding='utf-8') as json_file, open(output_path, 'wb') as bin_file:
        for line in tqdm(json_file, desc=f"Processing {base_name}", mininterval=1.0):
            try:
                data = json.loads(line)
                text = data['text']
                text_id = tokenizer.encode(text, add_special_tokens=False)
                text_id.append(tokenizer.eos_token_id)
                arr = np.array(text_id, dtype=np.uint16)
                bin_file.write(arr.tobytes())
            except Exception as e:
                print(f"Error processing line in {base_name}: {line}\nException: {e}")

    print(f"Data saved to {output_path}")

def process_open_web_math(data_path="/cephfs/shared/lichao/pretrain/data/Llemma/open-web-math",
                          output_dir="/cephfs/shared/lichao/pretrain/data/Llemma/open-web-math/llama",
                          tokenizer_name="huggyllama/llama-7b"):
    """
    处理 data_path 目录下的所有 JSON 文件，将其转换为二进制格式保存。

    Args:
    - data_path: 包含 JSON 文件的目录路径。
    - output_dir: 处理后数据保存的目录。
    - tokenizer_name: 使用的分词器名称。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 JSON 文件的路径
    json_files = [f for f in os.listdir(data_path) if f.endswith('.jsonl')]
    json_file_paths = [os.path.join(data_path, f) for f in json_files]
    
    print("json_file_paths=",json_file_paths)

    # 创建进程池
    num_processes = 10
    pool = multiprocessing.Pool(processes=num_processes)

    # 使用偏函数固定 tokenizer 和 output_dir 参数
    process_func = partial(process_file, tokenizer=tokenizer, output_dir=output_dir)

    # 使用进程池处理所有文件
    pool.map(process_func, json_file_paths)

    # 关闭进程池
    pool.close()
    pool.join()

    print("All files processed.")

if __name__ == "__main__":
    process_open_web_math()