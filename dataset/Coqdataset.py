from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import json
from torch.utils.data import Dataset
from jinja2 import Template
from dataset.utils import add_bos_eos, tokenize_str

from dataset.constants import GOAL_TOKEN_START, GOAL_TOKEN_END, CONTEXT_TOKEN_START, CONTEXT_TOKEN_END,\
    TACTIC_TOKEN_START , TACTIC_TOKEN_END, PARAMS_TOKEN_START, PARAMS_TOKEN_END,\
    BEFORE_STATE_TOKEN_START, BEFORE_STATE_TOKEN_END, AFTER_STATE_TOKEN_START, AFTER_STATE_TOKEN_END

COQ_TEMPLATE = Template(
    f"""{BEFORE_STATE_TOKEN_START}{CONTEXT_TOKEN_START}$before_context{CONTEXT_TOKEN_END}{GOAL_TOKEN_START}$goal{GOAL_TOKEN_END}{BEFORE_STATE_TOKEN_END}
    {TACTIC_TOKEN_START}$tactic{TACTIC_TOKEN_END}{PARAMS_TOKEN_START}$params{PARAMS_TOKEN_END}
    {AFTER_STATE_TOKEN_START}$after_states{AFTER_STATE_TOKEN_END}"""
)

AFTER_STATE_TEMPLATE = Template(
    f"""{CONTEXT_TOKEN_START}$context{CONTEXT_TOKEN_END}{GOAL_TOKEN_START}$goal{GOAL_TOKEN_END}"""
)

class Goal:
    def __init__(self, text: str, tokens: List[str], token_ids: str):
        self.text = text
        self.tokens = tokens
        self.token_ids = token_ids

class Tactic:
    def __init__(self, text: str, tokens: List[str], token_ids: str):
        self.text = text
        self.tokens = tokens
        self.token_ids = token_ids

class Proofstate:
    def __init__(self, data: Dict):
        self.before = self._parse_state(data.get('before', {}))
        self.after = [self._parse_state(state) for state in data.get('after', [])]
        self.tactic = self._parse_tactic(data.get('tactic', {}))
        self.file_id = data.get('file_id', '')

    def _parse_state(self, state: Dict) -> Dict:
        return {
            'contexts': state.get('contexts', []),
            'contextNames': state.get('contextNames', []),
            'goal': Goal(**state['goal']) if 'goal' in state else None
        }

    def _parse_tactic(self, tactic: Dict) -> Dict:
        return {
            'tactic': Tactic(**tactic['tactic']) if 'tactic' in tactic else None,
            'params': tactic.get('params', [])
        }

class CoqDataset(Dataset):
    def __init__(self, file_path: str):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(Proofstate(json.loads(line)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CoqDataCollator:
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        for item in batch:
            before_context = '\n'.join(item.before['contexts'])
            before_goal = item.before['goal'].text
            tactic = item.tactic['tactic'].text
            params = ' '.join(item.tactic['params']) if item.tactic['params'] else ""
            
            after_states = []
            for after_state in item.after:
                after_context = '\n'.join(after_state['contexts'])
                after_goal = after_state['goal'].text if after_state['goal'] else ""
                after_state_str = AFTER_STATE_TEMPLATE.substitute(
                    context=after_context,
                    goal=after_goal
                )
                after_states.append(after_state_str)
            
            after_states_str = "\n".join(after_states)

            formatted_input = COQ_TEMPLATE.substitute(
                before_context=before_context,
                goal=before_goal,
                tactic=tactic,
                params=params,
                after_states=after_states_str
            )
            input_id = tokenize_str(formatted_input, self.tokenizer, truncation=True)
            input_id = add_bos_eos(input_id, self.tokenizer)
            input_ids.append(torch.tensor(input_id))
            attention_mask = [1] * len(input_id)
            attention_masks.append(torch.tensor(attention_mask))
        
        batch_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
        
        batch_mask = torch.nn.utils.rnn.pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            )
        
        return {"input_ids": batch_ids, "attention_mask": batch_mask}
