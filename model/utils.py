import os.path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from utils.train import print_rank_0
from transformers import LlamaForCausalLM, LlamaConfig
from rho1.SLMforward import SelectiveAutoModelForCausalLM
from rho1.SLMentropy import SelectiveAutoModelEntropyForCausalLM

def init_config(pretrained_model_name_or_path: str):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    return config

def init_tokenizer(tokenizer_name: str, model_max_length: int, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=model_max_length, use_fast=True,**kwargs)
    return tokenizer

def init_from_pretrained(
    pretrained_dir: str,
    attn_implementation: Optional[str] = "flash_attention_2",
):

    config = AutoConfig.from_pretrained(
        os.path.join(pretrained_dir, "config"), attn_implementation=attn_implementation
    )

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_dir, "tokenizer"))

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_dir, 
        config=config, 
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16
    )

    return model, tokenizer, config

def init_slm_from_pretrained(
    pretrained_dir: str,
    attn_implementation: Optional[str] = "flash_attention_2",
):

    config = AutoConfig.from_pretrained(
        os.path.join(pretrained_dir, "config"), attn_implementation=attn_implementation
    )

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_dir, "tokenizer"))

    model = SelectiveAutoModelForCausalLM.from_pretrained(
        pretrained_dir, 
        config=config, 
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16
    )

    return model, tokenizer, config

def init_slm_entropy_from_pretrained(
    pretrained_dir: str,
    attn_implementation: Optional[str] = "flash_attention_2",
):

    config = AutoConfig.from_pretrained(
        os.path.join(pretrained_dir, "config"), attn_implementation=attn_implementation
    )

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_dir, "tokenizer"))

    model = SelectiveAutoModelEntropyForCausalLM.from_pretrained(
        pretrained_dir, 
        config=config, 
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16
    )

    return model, tokenizer, config

def init_from_pretrained(
    pretrained_dir: str,
    attn_implementation: Optional[str] = "flash_attention_2",
):

    config = AutoConfig.from_pretrained(
        os.path.join(pretrained_dir, "config"), attn_implementation=attn_implementation
    )

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_dir, "tokenizer"))

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_dir, 
        config=config, 
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16
    )

    return model, tokenizer, config

def init_model(attn_implementation: str = "flash_attention_2", model_max_length: int = 2048):
    config = LlamaConfig(num_hidden_layers=32, 
                         intermediate_size=8960, 
                         hidden_size=1536, 
                         num_attention_heads=12,
                         num_key_value_heads=2,
                         _attn_implementation=attn_implementation,
                         max_position_embeddings=model_max_length
                        )

    # 使用自定义配置初始化模型
    model = LlamaForCausalLM(config)

    return model,config

def init_llm_and_tokenizer(
    base_model_name: str,
    pretrained_dir: str = None,
    attn_implementation: str = "flash_attention_2",
    **kwargs,
):
    if pretrained_dir:
        return init_from_pretrained(pretrained_dir, attn_implementation)
    else:
        model,config = init_model()
        return model, init_tokenizer(base_model_name), config

# 统计模型参数量并输出
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {(total_params / 1024 / 1024 / 1024):.2f} B")

def save_model(
    model,
    config,
    tokenizer,
    output_dir,
):
    model.save_fp16_model(f"{output_dir}/fp16")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")
    config.save_pretrained(f"{output_dir}/config")
