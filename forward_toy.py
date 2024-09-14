import torch
import torch.nn as nn

class SelectiveLanguageModel(nn.Module):
    def __init__(self, reference_model, k_percent):
        super(SelectiveLanguageModel, self).__init__()
        self.reference_model = reference_model
        self.k_percent = k_percent
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets, attention_mask=None):
        # logits: [batch_size, seq_len, vocab_size]
        # targets: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]

        batch_size, seq_len, vocab_size = logits.size()

        current_loss = self.compute_loss(logits, targets, attention_mask)  # [batch_size, seq_len]

        with torch.no_grad():
            reference_loss = self.reference_model(logits, targets, attention_mask)  # [batch_size, seq_len]

        excess_loss = current_loss - reference_loss  # [batch_size, seq_len]

        valid_tokens = attention_mask.sum()
        k = int(valid_tokens * self.k_percent / 100)
        flat_excess_loss = excess_loss.view(-1)
        flat_attention_mask = attention_mask.view(-1)
        _, top_indices = torch.topk(flat_excess_loss.masked_fill(flat_attention_mask == 0, float('-inf')), k)
        
        slm_mask = torch.zeros_like(flat_excess_loss)
        slm_mask[top_indices] = 1
        slm_mask = slm_mask.view(batch_size, seq_len)

        mask = attention_mask * slm_mask

        slm_loss = (current_loss * mask).sum() / (mask.sum() + 1e-10)

        return slm_loss

    def compute_loss(self, logits, targets, attention_mask=None):
        # 计算每个token的交叉熵损失
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss = loss.view_as(targets)

        if attention_mask:
            loss = loss * attention_mask

        return loss