from typing import List

import torch
import transformers

def tokenize_str(
    string: str,
    tokenizer: transformers.PreTrainedTokenizer,
    add_special_tokens: bool = False,
    truncation: bool = False,
) -> List[int]:
    input_ids = tokenizer(
        string,
        add_special_tokens=add_special_tokens,
        max_length=tokenizer.model_max_length,
        truncation=truncation,
    )["input_ids"]
    return input_ids

def add_bos_eos(input_ids: List[int], tokenizer: transformers.PreTrainedTokenizer) -> List[int]:
    return [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]

def padding_tensor_list(
    tensor_list: List[torch.Tensor], padding_value: float, padding="right"
) -> torch.Tensor:
    max_context_count = max([i.shape[0] for i in tensor_list])
    max_context_len = max([i.shape[1] for i in tensor_list])
    padded_tensor = torch.full((len(tensor_list), max_context_count, max_context_len), padding_value)

    for i, tensor in enumerate(tensor_list):
        context_count, context_len = tensor.shape
        if padding == "right":
            padded_tensor[i, :context_count, :context_len] = tensor
        elif padding == "left":
            padded_tensor[i, :context_count, -context_len:] = tensor
        else:
            raise Exception("padding must be 'right' or 'left'")

    return padded_tensor
