import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class SelectiveAutoModelForCausalLM(AutoModelForCausalLM):
    def __init__(self, config, k_percent: float = 50.0):
        super().__init__(config)
        self.k_percent = k_percent
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=None,
                                **kwargs
                                )
        logits = outputs.logits # [batch_size, seq_len, vocab_size]

        selective_loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous() if attention_mask is not None else None
            current_loss = self.compute_loss(shift_logits, shift_labels, shift_attention_mask)  # [batch_size, seq_len - 1]

        with torch.no_grad():
            reference_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs
            )
        reference_logits = reference_outputs.logits
        shift_reference_logits = reference_logits[..., :-1, :].contiguous()
        reference_loss = self.compute_loss(shift_reference_logits, shift_labels, shift_attention_mask)  # [batch_size, seq_len - 1]

        excess_loss = current_loss - reference_loss # [batch_size, seq_len - 1]

        valid_tokens = attention_mask.sum(dim=-1) # [batch_size]

        k = (valid_tokens.float() * self.k_percent / 100.0).clamp(min=1).long() # [batch_size]

        selective_loss = []

        for i in range(logits.size(0)): # 遍历 batch
            per_example_excess_loss = excess_loss[i].clone() # [seq_len - 1]
            per_example_attention_mask = attention_mask[i] # [seq_len - 1]

            per_example_excess_loss = per_example_excess_loss.masked_fill(per_example_attention_mask == 0, float('-inf'))

            _, top_indices = torch.topk(per_example_excess_loss, k[i], largest=True)

            slm_mask = torch.zeros_like(per_example_excess_loss)
            slm_mask.scatter_(0, top_indices, 1)

            per_example_selective_loss = (current_loss[i] * slm_mask).sum() / (slm_mask.sum() + 1e-10)
            selective_loss.append(per_example_selective_loss)

        selective_loss = torch.stack(selective_loss) # [batch_size]

        outputs.loss = selective_loss.mean()

        return outputs

    def compute_loss(self, logits, targets, attention_mask=None):
        """
        计算每个 token 的交叉熵损失。

        Args:
        logits (torch.Tensor): 模型输出的 logits，形状 [batch_size, seq_len, vocab_size]。
        targets (torch.Tensor): 标签，形状 [batch_size, seq_len]。
        attention_mask (torch.Tensor, optional): 注意力掩码，形状 [batch_size, seq_len]。

        Returns:
        torch.Tensor: 每个 token 的损失，形状 [batch_size, seq_len]。
        """
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1)) # [batch_size * seq_len]
        loss = loss.view_as(targets) # [batch_size, seq_len]

        if attention_mask is not None:
            loss = loss * attention_mask # 只计算有效 token 的损失

        return loss