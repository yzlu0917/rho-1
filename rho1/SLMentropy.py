import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class SelectiveAutoModelEntropyForCausalLM(AutoModelForCausalLM):
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
        reference_info_entropy = self.compute_info_entropy(shift_reference_logits)  # [batch_size, seq_len - 1]
        
        selection_criterion = current_loss - reference_info_entropy
        
        valid_tokens = attention_mask.sum(dim=-1) # [batch_size]

        k = (valid_tokens.float() * self.k_percent / 100.0).clamp(min=1).long() # [batch_size]

        selective_loss = []

        for i in range(logits.size(0)): # Iterate over batch
            per_example_criterion  = selection_criterion[i].clone() # [seq_len - 1]
            per_example_attention_mask = attention_mask[i, 1:] if attention_mask is not None else None # [seq_len - 1]

            if per_example_attention_mask is not None:
                per_example_criterion = per_example_criterion.masked_fill(per_example_attention_mask == 0, float('-inf'))

            _, top_indices = torch.topk(per_example_criterion, k[i], largest=True)

            slm_mask = torch.zeros_like(per_example_criterion)
            slm_mask.scatter_(0, top_indices, 1)

            per_example_selective_loss = (current_loss[i] * slm_mask).sum() / (slm_mask.sum() + 1e-10)
            selective_loss.append(per_example_selective_loss)

        selective_loss = torch.stack(selective_loss) # [batch_size]

        outputs.loss = selective_loss.mean()

        return outputs

    def compute_loss(self, logits, targets, attention_mask=None):
        """
        Compute cross-entropy loss for each token.

        Args:
        logits (torch.Tensor): Model output logits, shape [batch_size, seq_len, vocab_size].
        targets (torch.Tensor): Labels, shape [batch_size, seq_len].
        attention_mask (torch.Tensor, optional): Attention mask, shape [batch_size, seq_len].

        Returns:
        torch.Tensor: Loss for each token, shape [batch_size, seq_len].
        """
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1)) # [batch_size * seq_len]
        loss = loss.view_as(targets) # [batch_size, seq_len]

        if attention_mask is not None:
            loss = loss * attention_mask

        return loss
    
    def compute_info_entropy(self, logits):
        """
        Compute the information entropy for each token.
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy