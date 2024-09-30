import os.path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from utils.train import print_rank_0
from transformers import LlamaForCausalLM, LlamaConfig
from rho1.SLMforward import SelectiveAutoModelForCausalLM
from rho1.SLMentropy import SelectiveAutoModelEntropyForCausalLM
from model.tokenization import add_special_tokens_to_tokenizer

def model_embedding_resize(model, tokenizer, num_new_tokens):
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def add_special_tokens_to_to_tokenizer(tokenizer, model):
    num_new_tokens = add_special_tokens_to_tokenizer(tokenizer)
    print('================')
    print(num_new_tokens)
    model_embedding_resize(model, tokenizer, num_new_tokens)

def init_config(pretrained_model_name_or_path: str,
                attn_implementation: Optional[str] = "flash_attention_2"):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                        attn_implementation=attn_implementation)
    return config
def update_config(config):
    print_rank_0("Config:", wrap=True)
    print_rank_0(config)
    
    config.pad_token_id = config.vocab_size
    config.global_token_start_id = config.vocab_size + 1
    config.global_token_end_id = config.vocab_size + 2
    config.context_token_start_id = config.vocab_size +3
    config.context_token_end_id = config.vocab_size + 4
    config.tactic_token_start_id = config.vocab_size +5
    config.tactic_token_end_id = config.vocab_size + 6
    config.params_token_start_id = config.vocab_size + 7
    config.params_token_end_id = config.vocab_size + 8
    config.before_state_token_start_id = config.vocab_size + 9
    config.before_state_token_end_id = config.vocab_size + 10
    config.after_state_token_start_id = config.vocab_size + 11
    config.after_state_token_end_id = config.vocab_size + 12
    config.vocab_size += 13
    return config

def init_tokenizer(tokenizer_name: str, model_max_length: int, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=model_max_length, use_fast=True,**kwargs)
    return tokenizer

def init_from_pretrained(
    pretrained_dir: str,
    attn_implementation: Optional[str] = "flash_attention_2",
):

    config = AutoConfig.from_pretrained(
        pretrained_dir, attn_implementation=attn_implementation
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)

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
        pretrained_dir, attn_implementation=attn_implementation
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)

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
    model_max_length: int = 2048
):

    config = init_config(pretrained_dir, attn_implementation)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_dir,
        config=config,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16
    )
    print_rank_0("Model:", wrap=True)
    print_rank_0(model)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir,model_max_length=model_max_length)
    add_special_tokens_to_to_tokenizer(tokenizer, model)

    print_rank_0("Tokenizer:", wrap=True)
    print_rank_0(tokenizer)
    
    config = update_config(config)

    model.config.pad_token_id = config.pad_token_id
    model.config.global_token_start_id = config.global_token_start_id
    model.config.global_token_end_id = config.global_token_end_id
    model.config.context_token_start_id = config.context_token_start_id
    model.config.context_token_end_id = config.context_token_end_id
    model.config.tactic_token_start_id = config.tactic_token_start_id
    model.config.tactic_token_end_id = config.tactic_token_end_id
    model.config.params_token_start_id = config.params_token_start_id
    model.config.params_token_end_id = config.params_token_end_id
    model.config.before_state_token_start_id = config.before_state_token_start_id
    model.config.before_state_token_end_id = config.before_state_token_end_id
    model.config.after_state_token_start_id = config.after_state_token_start_id
    model.config.after_state_token_end_id = config.after_state_token_end_id
    model.config.vocab_size = config.vocab_size

    print_rank_0("New Config:", wrap=True)
    print_rank_0(config)

    print_rank_0("New model Config:", wrap=True)
    print_rank_0(model.config)

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
