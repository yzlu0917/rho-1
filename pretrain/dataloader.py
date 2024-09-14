import os
import random
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.load_file import find_files_with_suffix
from utils.logger import print_rank_0

class PretrainDataset(Dataset):
    """加载数据
    这里保存了多个文件到sample的映射, 以便于节约内存
    """

    def __init__(self, data_path, max_length: int = 4096):
        self.data = []
        self.index_map = {}
        self.token_size, self.smp_size = 0, 0
        filenames = find_files_with_suffix(data_path, ".bin")
        
        print_rank_0(f"files={filenames}")
        
        all_samples = []
        
        for fi, filename in enumerate(filenames):
            with open(filename, 'r') as f:
                nbytes = f.seek(0, 2)
                flen = f.tell() // np.dtype('uint16').itemsize
            self.token_size += flen
            
            num_samples = flen // max_length
            all_samples.extend([(fi, i) for i in range(num_samples)])

            self.data.append(
                np.memmap(filename, dtype=np.dtype('uint16'), shape=(flen // max_length, max_length)))
        
        # Shuffle all samples
        random.shuffle(all_samples)
        
        # Create shuffled index map
        self.index_map = {i: sample for i, sample in enumerate(all_samples)}
        self.smp_size = len(all_samples)
        
        print_rank_0(f'token_size: {self.token_size}, smp_size: {self.smp_size}')

    def __len__(self):
        return self.smp_size

    def __getitem__(self, index: int):
        fi, i = self.index_map[index]
        sample = self.data[fi][i]
        x = np.array(sample).astype(np.int64)
        
        input_ids = torch.from_numpy(x)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }