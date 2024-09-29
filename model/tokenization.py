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