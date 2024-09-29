import random
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import transformers


@dataclass
class PretrainArguments(transformers.TrainingArguments):
    # Model arguments
    model_name_or_path: Optional[str] = field(default="EleutherAI/llemma_7b")
    dropout: float = field(default=0, metadata={"help": "model dropout"})
    zero_stage: int = field(default=1, metadata={"help": "zero stage"})
    offload_adam: bool = field(default=False, metadata={"help": "offload adam parameters to cpu"})
    offload_params: bool = field(default=False, metadata={"help": "offload model parameters  to cpu"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing"})
    
    # Data arguments
    data_path: str = field(default="/cephfs/shared/lichao/pretrain/data/pretrain_data/train",
                           metadata={"help": "Path to the training data."})

    eval_path: str = field(default="/cephfs/shared/lichao/pretrain/data/pretrain_data/test",
                           metadata={"help": "Path to the evaluating data."})
    
    # Training arguments
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Path to the huggingface hub."})
    model_max_length: int = field(default=2048, metadata={
        "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    overwrite_output_dir: bool = field(default=True)
    learning_rate: float = field(default=1e-5, metadata={"help": "init learning rate"})
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "attention implementation"})
    fp32_loss: bool = field(default=False, metadata={"help": "whether calculate loss in fp32"})
    scheduler: str = field(default="cosine", metadata={"help": "The scheduler type to use, default to consine"})
    warmup: float = field(default=0.1, metadata={"help": "Warmup ratio for scheduler"})
    
    # wandb
    wandb_enabled: bool = field(default=False, metadata={"help": "whether use wandb"})
    wandb_project_name: str = field(default="pretrain", metadata={"help": "wandb project name"})
    

    def __str__(self):
        # 使用dataclasses.asdict()将所有字段转换为字典
        params_dict = asdict(self)
        # 创建表示参数的字符串
        params_str = '\n'.join(f"{k}: {v}" for k, v in params_dict.items())
        return params_str
