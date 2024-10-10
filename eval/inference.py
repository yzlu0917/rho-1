import json
import random

import torch
from vllm import LLM
from vllm import SamplingParams
from utils.load_file import load_json_file
from utils.train import set_random_seed
from dataset.Coqdataset import EVAL_TEMPLATE, CoqDataset
from dataset.constants import PARAMS_TOKEN_END
set_random_seed(42)

data_path = '/cephfs/shared/yanghanbin/data/coq/test_data_proof_state_transition_outcomes_v6.jsonl'
output_path = '/cephfs/shared/luyanzhen/data/coq/test_data_proof_state_transition_outcomes_v6_result.jsonl'
model_path = 'xxx'

tensor_parallel_size = 1

dataset = CoqDataset(data_path)

model = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, dtype=torch.bfloat16, tokenizer=model_path)

prompts = []

raw_data = load_json_file(data_path)
indices = random.sample(range(len(dataset)), 10000)
original_data = [raw_data[i] for i in indices]

for idx in indices:
    item = dataset[idx]
    before_context = '\n'.join(ctx.text for ctx in item.before['contexts'])
    before_goal = item.before['goal'].text
    tactic = item.tactic['tactic'].text
    params = ' '.join(param.text for param in item.tactic['params']) if item.tactic['params'] else ""
    prompt = EVAL_TEMPLATE.substitute(
        context = before_context,
        goal = before_goal
    )    
    prompts.append(prompt)
    
    # after_states = []
    # for after_state in item.after:
    #     after_context = '\n'.join(ctx.text for ctx in after_state['contexts'])
    #     after_goal = after_state['goal'].text if after_state['goal'] else ""
    #     after_state_str = AFTER_STATE_TEMPLATE.substitute(
    #         context=after_context,
    #         goal=after_goal
    #     )
    #     after_states.append(after_state_str)
    # after_states_str = "\n".join(after_states)
    
stop = [PARAMS_TOKEN_END]
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_new_tokens=50, stop=stop)
outputs = model.generate(prompts, sampling_params)

with open(output_path, "w", encoding='utf-8') as f:
    for raw_item, output in zip(original_data, outputs):
        raw_item['output'] = output.outputs[0].text
        json.dump(raw_item, f, ensure_ascii=False)
        f.write('\n')
