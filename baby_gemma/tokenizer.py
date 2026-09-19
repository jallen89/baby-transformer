import torch


class SimpleTokenizer:

    def __init__(self):
        self.stoi = dict()
        self.itos = dict()
        self.unknown = '<UNK>'
        self.vocab_size = 0

    def construct(self, text: str):
        unique_words = sorted(list(set(text.split(' '))))
        self.words = text.split(' ')
        self.vocab_size = len(unique_words)
        self.stoi = {w: i for (i, w) in enumerate(unique_words)}
        self.itos = {i: w for (i, w) in enumerate(unique_words)}

    def encode(self, text: str):
        out = []
        for word in text.split(' '):
            out.append(self.stoi[word])
        return torch.tensor(out, dtype=torch.float)

    def decode(self, tensor: torch.Tensor):
        token_tensor = tensor.detach().cpu()

        if token_tensor.dim() == 0:
            return self.itos.get(int(token_tensor.item()), self.unknown)

        if token_tensor.dim() == 1:
            return ' '.join([self.itos.get(int(x), self.unknown) for x in token_tensor.tolist()])

        if token_tensor.dim() == 2:
            return [
                " ".join(self.itos.get(int(idx), self.unknown) for idx in row.tolist())
                for row in token_tensor
            ]

