import torch
import math 
import torch.nn as nn 

class AttentionHead(nn.Module):

    def __init__(self, d_model, max_seq_len=512, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        # in and out set o d_model since this is a single head. 
        self.Q = torch.nn.Linear(d_model, d_model)
        self.K = torch.nn.Linear(d_model, d_model)
        self.V = torch.nn.Linear(d_model, d_model)

        mask = torch.triu(
            torch.ones((max_seq_len, max_seq_len), dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer('mask', mask)

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        seq_len = x.shape[-2] #... x seq_len x dim_model
        qk = q @ k.transpose(-2, -1)
        qk_masked = qk.masked_fill(self.mask[:seq_len, :seq_len], -torch.inf)
        o = torch.softmax(qk_masked / math.sqrt(self.d_model), dim=-1) @ v
        return o