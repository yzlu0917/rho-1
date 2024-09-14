import logging
import os
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from transformers import AutoTokenizer
import queue

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level="DEBUG")

def process_batch(batch, tokenizer, file_path):
    try:
        with open(file_path, "ab") as f:
            for item in batch:
                text = item['text']
                encoded = tokenizer.encode(text, add_special_tokens=False)
                encoded.append(tokenizer.eos_token_id)
                buffer = np.array(encoded, dtype=np.uint16)
                f.write(buffer.tobytes())
        logger.debug(f"Processed batch of size {len(batch)}")
        return file_path
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return file_path

def batch_generator(dataset, batch_size=1000):
    batch = []
    for item in dataset:
        if item['meta']['redpajama_set_name'] in ['RedPajamaC4', 'RedPajamaCommonCrawl']: 
            if random.random() < 0.15:
                batch.append(item)
        else:
            batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def process_split(split_name, tokenizer, data_path, output_dir):
    dataset = load_dataset(data_path, split=split_name, streaming=True)
    batch_gen = batch_generator(dataset)

    num_processes = 40
    output_paths = [f"{output_dir}/{split_name}/chunk_{i}.bin" for i in range(num_processes)]
    
    for path in output_paths:
        if os.path.exists(path):
            os.remove(path)
    
    available_files = queue.Queue()
    for path in output_paths:
        available_files.put(path)

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = set()
        total_batches = 0
        pbar = tqdm(desc=f"Processing {split_name}", unit="batch")

        for batch in batch_gen:
            while len(futures) >= num_processes:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    file_path = future.result()
                    available_files.put(file_path)
                    pbar.update(1)

            if available_files.empty():
                # Wait for any future to complete if no files are available
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    file_path = future.result()
                    available_files.put(file_path)
                    pbar.update(1)

            file_path = available_files.get()
            future = executor.submit(process_batch, batch, tokenizer, file_path)
            futures.add(future)
            total_batches += 1

        # 等待所有剩余的任务完成
        for future in as_completed(futures):
            future.result()
            pbar.update(1)

        pbar.close()

    logger.info(f"All files for {split_name} have been processed and saved.")

def process_slim_pajama(data_path="/cephfs/shared/hf_cache/datasets/SlimPajama-627B",
                        output_dir="/cephfs/shared/lichao/pretrain/data/filted_slim_pajama_627b",
                        tokenizer_name="huggyllama/llama-7b"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    logger.info(f"Using Fast Tokenizer: {tokenizer.is_fast}")
    os.makedirs(output_dir, exist_ok=True)
    splits = ['train']
    for split in splits:
        process_split(split, tokenizer, data_path, output_dir)

if __name__ == "__main__":
    process_slim_pajama()