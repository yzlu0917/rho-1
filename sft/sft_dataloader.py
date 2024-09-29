import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from utils.logger import is_rank_0, print_rank_0
from utils.utils import pad_sequences
  
IGNORE_INDEX = -100 

def apply_chat_template(messages, bos_token="<s>", eos_token="</s>", add_generation_prompt=False):
    template = ""
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        template += f"{bos_token}{role}\n{content}{eos_token}\n"
    
    # 如果add_generation_prompt为True，添加生成提示
    if add_generation_prompt:
        template += f"{bos_token}assistant\n"
    
    return template

def preprocess_multiturn_conversation(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    add_generation_prompt: bool = False
) -> Dict[str, List[int]]:
    input_ids = []
    labels = [] if not add_generation_prompt else None
    bos_token = tokenizer.bos_token or "<s>"
    eos_token = tokenizer.eos_token or "</s>"

    for message in messages:
        role = message["role"]
        content = message["content"]

        role_ids = tokenizer.encode(f"{bos_token}{role}\n", add_special_tokens=False)
        
        content_ids = tokenizer.encode(f"{content}{eos_token}\n", add_special_tokens=False)
        
        input_ids.extend(role_ids + content_ids)
        if not add_generation_prompt:
            if role == "assistant":
                labels.extend([IGNORE_INDEX] * len(role_ids) + content_ids)
            else:
                labels.extend([IGNORE_INDEX] * (len(role_ids) + len(content_ids)))

    if add_generation_prompt:
        prompt_ids = tokenizer.encode(f"{bos_token}assistant\n", add_special_tokens=False)
        input_ids.extend(prompt_ids)

    # left truncation
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        if labels is not None:
            labels = labels[-max_length:]

    attention_mask = [1] * len(input_ids)

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if labels is not None:
        result["labels"] = labels

    return result

def create_batch(
    conversations: List[List[Dict[str, str]]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    add_generation_prompt: bool = False
) -> Dict[str, torch.Tensor]:
    batch_inputs = []
    
    for conversation in conversations:
        inputs = preprocess_multiturn_conversation(conversation, tokenizer, max_length, add_generation_prompt)
        batch_inputs.append(inputs)
    
    max_length_in_batch = max(len(inputs["input_ids"]) for inputs in batch_inputs)
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = None if add_generation_prompt else []
    
    for inputs in batch_inputs:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        padding_length = max_length_in_batch - len(input_ids)
        
        # left padding
        input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
        attention_mask = [0] * padding_length + attention_mask
        
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        
        if not add_generation_prompt:
            labels = inputs["labels"]
            labels = [IGNORE_INDEX] * padding_length + labels
            batch_labels.append(labels)
    
    result = {
        "input_ids": torch.tensor(batch_input_ids),
        "attention_mask": torch.tensor(batch_attention_mask),
    }
    if batch_labels is not None:
        result["labels"] = torch.tensor(batch_labels)
    
    return result


class SFTDataset(Dataset):
    def __init__(self, file_path, tokenizer: PreTrainedTokenizer, max_length: int):
        self.data = []
        
        # 读取 JSONL 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="loading data", disable=not is_rank_0()):
                item = json.loads(line)
                # prompt = apply_chat_template(item)
                # if len(tokenizer.encode(prompt, add_special_tokens=False)) < max_length:
                #     self.data.append(item)
                self.data.append(item)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


class ChatDataCollectFunctor:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __call__(self, batch: List[dict]):
        
        # # prompts = [apply_chat_template(ms, bos_token=self.tokenizer.bos_token, eos_token=self.tokenizer.eos_token) for ms in batch]
        # prompts = [self.tokenizer.apply_chat_template(ms, tokenize=False) + self.tokenizer.eos_token for ms in batch]

        # # print_rank_0(prompts[0])

        # # 使用 tokenizer 对 batch 进行标记化和填充
        # prompt_encodings = self.tokenizer(prompts,
        #                                   add_special_tokens=False,
        #                                   truncation=True,
        #                                   padding=True,
        #                                   max_length=self.max_length,
        #                                   return_tensors="pt"
        #                                   )

        # # 返回处理过的批次
        # return {
        #     'input_ids': prompt_encodings["input_ids"],
        #     'attention_mask': prompt_encodings["attention_mask"],
        #     'labels': prompt_encodings["input_ids"]
        # }
        
        return create_batch(conversations=batch, tokenizer=self.tokenizer, max_length=self.max_length)

INSTRUCTION_KEY = "instruction"
RESPONSE_KEY = "response"   
        
class QADataCollectFunctor:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.max_length = max_length
        self.tokenizer = tokenizer
    
    def transform_array(self, arr: torch.LongTensor):
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] == 1:
                if i < len(arr) - 1:
                    arr[i+1] = 1
                break
        return arr

    def __call__(self, batch: List[dict]):
        prompts = [f"[INST] {item[INSTRUCTION_KEY]} [/INST] " for item in batch]
        response = [item[RESPONSE_KEY] for item in batch]

        # 使用 tokenizer 对 batch 进行标记化和填充
        prompt_encodings = self.tokenizer(prompts, add_special_tokens=True)
        resp_encodings = self.tokenizer(response, add_special_tokens=False)

        input_ids = [prompt + answer + [self.tokenizer.eos_token_id] for prompt, answer in
                     zip(prompt_encodings['input_ids'], resp_encodings['input_ids'])]

        input_max_length = min(self.max_length, max(map(len, input_ids)))
        
        # 目前支持left padding和left 
        truncated_input_ids = [seq[-input_max_length:] for seq in input_ids]
        padded_input_ids = pad_sequences(truncated_input_ids, self.tokenizer.pad_token_id, padding="left",
                                  pad_to=input_max_length)

        input_ids = torch.tensor(padded_input_ids)
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        
        # unmask eos token 
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            attention_mask = torch.stack([self.transform_array(mask) for mask in attention_mask])

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        # 目前支持left padding和left truncation
        # len(ids) + 1 因为结尾补了一个eos token
        for i, length in enumerate([len(ids) + 1 for ids in resp_encodings['input_ids']]):
            labels[i, :input_max_length - length] = -100

        # 返回处理过的批次
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }