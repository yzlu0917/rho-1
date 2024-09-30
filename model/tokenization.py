from typing import Any, Dict
import torch
from transformers import AutoTokenizer, LlamaTokenizer

from dataset.constants import GOAL_TOKEN_START, GOAL_TOKEN_END, CONTEXT_TOKEN_START, CONTEXT_TOKEN_END,\
    TACTIC_TOKEN_START , TACTIC_TOKEN_END, PARAMS_TOKEN_START, PARAMS_TOKEN_END,\
    BEFORE_STATE_TOKEN_START, BEFORE_STATE_TOKEN_END, AFTER_STATE_TOKEN_START, AFTER_STATE_TOKEN_END, DEFAULT_PAD_TOKEN


def init_tokenizer(tokenizer_name: str, model_max_length: int, **kwargs) -> LlamaTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=model_max_length, **kwargs)
    tokenizer.padding_side = "left"
    return tokenizer


def add_special_tokens_to_tokenizer(tokenizer):
    special_token_dict = {
        "pad_token": DEFAULT_PAD_TOKEN,
        "gls": GOAL_TOKEN_START,
        "gle": GOAL_TOKEN_END,
        "cts": CONTEXT_TOKEN_START,
        "cte": CONTEXT_TOKEN_END,
        "tcs": TACTIC_TOKEN_START,
        "tce": TACTIC_TOKEN_END,
        "pms": PARAMS_TOKEN_START,
        "pme": PARAMS_TOKEN_END,
        "bss": BEFORE_STATE_TOKEN_START,
        "bse": BEFORE_STATE_TOKEN_END,
        "ass": AFTER_STATE_TOKEN_START,
        "ase": AFTER_STATE_TOKEN_END
    }
    for key in special_token_dict.keys():
        if key not in tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append(key)

    num_new_tokens = tokenizer.add_special_tokens(special_token_dict)
    for key, value in special_token_dict.items():
        setattr(tokenizer, f"_{key}", getattr(tokenizer, key))
        setattr(tokenizer, f"{key}_id", tokenizer.convert_tokens_to_ids(value))
    return num_new_tokens

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    logpy = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logpy


def get_category_distribution_entropy(bsz, logits):
    """
    Compute category distribution entropy
    """
    logits_distribution = torch.distributions.categorical.Categorical(logits=logits.reshape(-1, logits.size(-1)))
    ent = logits_distribution.entropy().reshape(bsz, -1)
    return ent


def top_p_logits(logits, topp=0.9, filter_value=0, min_topk=1):
    """
    Filter a distribution of logits using nucleus (top-p) filtering
    https://github.com/OpenLMLab/MOSS/blob/e088f438d1a95d424c6dffef0d73134ebe62cb72/models_jittor/generation.py#L146
    """
    cum_logits = logits.clone()
    if topp > 0:
        logits_sorted, inds = torch.sort(logits, dim=-1, descending=True)
        mask = (logits_sorted.cumsum(dim=-1) - logits_sorted) >= topp
        mask[:, :min_topk] = False
        # Remove tokens with cumulative top_p above the threshold
        mask = torch.zeros_like(mask).to(torch.bool).scatter_(dim=-1, index=inds, src=mask)
        cum_logits[mask] = filter_value
        cum_logits.div_(cum_logits.sum(dim=-1, keepdim=True))
        
    return cum_logits


def clean_dict(d: Dict[str, Any]) -> None:
    """
    清理字典，删除所有项目。

    参数:
    d (Dict[str, Any]): 需要清理的字典

    返回:
    None
    """
    for key in list(d.keys()):
        del d[key]
    del d