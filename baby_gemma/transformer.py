from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from layers.norm import RMSNorm
from layers.attention import AttentionHead
from layers.base import FFN

class Transformer(nn.Module, ABC):
    def __init__(self, seq_len: int, vocab_size: int, d_model: int):


        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model

    @abstractmethod
    def forward(self, x):
        pass

    def generate(self, tokens, max_seq_len=50, temperature=1.0, top_k=40):

        device = next(self.parameters()).device
        tokens = tokens.to(device)

        for _ in range(max_seq_len):
            logits = self(tokens)
            logits = logits[:,-1,:]


            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, top_k)
                    min_value = values[:, [-1]]
                    logits[logits < min_value] = -torch.inf 


                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)    

            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens 


class BabyTransformer(Transformer):
    '''
    Embeddings, no position information, 
    single-head scaled dot-product attention
    '''

    def __init__(self, seq_len: int, vocab_size: int, d_model: int,
                 dropout: int =0.1):
        super().__init__(seq_len, vocab_size, d_model)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.prenorm_1 = RMSNorm(d_model)
        self.head = AttentionHead(d_model, seq_len, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.prenorm_2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_model*4, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.prenorm_3 = RMSNorm(d_model)
        self.linear = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.emb_dropout(emb)
        x_norm = self.prenorm_1(emb)
        head_o = emb + self.attention_dropout(self.head(x_norm))
        head_o_norm = self.prenorm_2(head_o)
        ffn_o = head_o + self.ffn_dropout(self.ffn(head_o_norm))
        ffn_o_norm = self.prenorm_3(ffn_o)
        logits = self.linear(ffn_o_norm) 
        return logits 