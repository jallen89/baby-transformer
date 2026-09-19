from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class Transformer(nn.Module, ABC):
    def __init__(self, itos: dict, stoi: dict, seq_len: int, vocab_size: int, d_model: int):
        self.itos = itos
        self.stoi = stoi 

        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model

    @abstractmethod
    def forward(self, x):
        pass
        

    def tokens_to_words(self, seq):
        output = ""
        for i in seq[0]:
            s = self.itos[i.item()]
            output = output + " " + s 
        return output

    def context_to_tokens(self, context):
        tokens = []
        for w in context.split(' '):
            idx = self.stoi[w]
            tokens.append(idx)
        return torch.tensor(tokens).unsqueeze(0)


    def generate(self, max_seq_len=50, temperature=1.0, top_k=40, context="We are accounted poor"):

        device = next(self.parameters()).device
        tokens = self.context_to_tokens(context).to(device)


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


        output = self.tokens_to_words(tokens)
        return output