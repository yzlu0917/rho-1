from dataclasses import dataclass
import os
from typing import List, Dict, Optional, Union

import torch
import json
from torch.utils.data import Dataset
from utils.train import print_rank_0

from string import Template
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

EVAL_TEMPLATE =  Template(
    f"""{CONTEXT_TOKEN_START}$context{CONTEXT_TOKEN_END}{GOAL_TOKEN_START}$goal{GOAL_TOKEN_END}{TACTIC_TOKEN_START}"""
)

# class TextTokenIds:
#     def __init__(self, data: Dict):
#         self.text = data.get('text', '')
#         self.tokens = data.get('tokens', [])
#         self.token_ids = data.get('token_ids', '')

# class Goal(TextTokenIds):
#     pass

# class Tactic(TextTokenIds):
#     pass

# class Context(TextTokenIds):
#     pass

# class ContextName(TextTokenIds):
#     pass

# class Param(TextTokenIds):
#     pass

# class Proofstate:
#     def __init__(self, data: Dict):
#         self.before = self._parse_state(data.get('before', {}))
#         self.after = [self._parse_state(state) for state in data.get('after', [])]
#         self.tactic = self._parse_tactic(data.get('tactic', {}))
#         self.file_id = data.get('file_id', '')

#     def _parse_state(self, state: Dict) -> Dict:
#         return {
#             'contexts': [Context(ctx) for ctx in state.get('contexts', [])],
#             'contextNames': [ContextName(name) for name in state.get('contextNames', [])],
#             'goal': Goal(state['goal']) if 'goal' in state else None
#         }

#     def _parse_tactic(self, tactic: Dict) -> Dict:
#         return {
#             'tactic': Tactic(tactic['tactic']) if 'tactic' in tactic else None,
#             'params': [Param(param) for param in tactic.get('params', [])]
#         }

class TextOnly:
    def __init__(self, text: str):
        self.text = text
        
    def __repr__(self):
        return f"{self.__class__.__name__}('{self.text}')"
    
class Goal(TextOnly):
    pass

class Tactic(TextOnly):
    pass

class Context(TextOnly):
    pass

class ContextName(TextOnly):
    pass

class Param(TextOnly):
    pass

class Proofstate:
    def __init__(self, data: Dict):
        self.before = self._parse_state(data.get('before', {}))
        self.after = [self._parse_state(state) for state in data.get('after', [])]
        self.tactic = self._parse_tactic(data.get('tactic', {}))
        self.file_id = data.get('file_id', '')

    def _parse_state(self, state: Dict) -> Dict:
        return {
            'contexts': [Context(ctx.get('text', '')) for ctx in state.get('contexts', [])],
            # 'contextNames': [ContextName(name.get('text', '')) for name in state.get('contextNames', [])],
            'goal': Goal(state['goal'].get('text', '')) if 'goal' in state else None
        }

    def _parse_tactic(self, tactic: Dict) -> Dict:
        return {
            'tactic': Tactic(tactic['tactic'].get('text', '')) if 'tactic' in tactic else None,
            'params': [Param(param.get('text', '')) for param in tactic.get('params', [])]
        }
        
    def __repr__(self):
        return f"Proofstate(before={self.before}, after={self.after}, tactic={self.tactic}, file_id='{self.file_id}')"

class CoqDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, List[str]],
        data_limit: int = None
    ):
        super().__init__()
        self.data_files = self._get_data_files(data_path)
        self.data_limit = data_limit
        self.total_lines = self._count_lines_per_file()
        self.data = self._load_all_data()
        if self.data_limit:
            self.data = self.data[:data_limit]
            # self.total_lines = min(self.total_lines, self.data_limit)
            
    def _get_data_files(self, data_path):
        if isinstance(data_path, str):
            if os.path.isdir(data_path):
                return [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jsonl')]
            elif os.path.isfile(data_path) and data_path.endswith('.jsonl'):
                return [data_path]
        elif isinstance(data_path, list):
            return [f for f in data_path if os.path.isfile(f) and f.endswith('.jsonl')]
        raise ValueError("Invalid data_path. Must be a .jsonl file, a directory containing .jsonl files, or a list of .jsonl files.")

    def _count_lines_per_file(self):
        print_rank_0('count_line', wrap=True)
        line_counts = []
        for f in self.data_files:
            with open(f, 'r') as file:
                line_counts.append(sum(1 for _ in file))
        return line_counts
    
    def __len__(self):
        return len(self.data)
    
    def _load_all_data(self):
        all_data = []
        print_rank_0('load all data', wrap=True)
        # print(self.data_files)
        for file_path in self.data_files:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    all_data.append(Proofstate(data))
        return all_data
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class CoqDataCollator:
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        for item in batch:
            before_context = '\n'.join(ctx.text for ctx in item.before['contexts'])
            before_goal = item.before['goal'].text
            tactic = item.tactic['tactic'].text
            params = ' '.join(param.text for param in item.tactic['params']) if item.tactic['params'] else ""
            
            after_states = []
            for after_state in item.after:
                after_context = '\n'.join(ctx.text for ctx in after_state['contexts'])
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
                attention_masks,
                batch_first=True,
                padding_value=0,
            )
        
        labels = batch_ids.clone()
        labels[batch_mask == 0] = -100
        
        return {"input_ids": batch_ids, "attention_mask": batch_mask, "labels": labels}
