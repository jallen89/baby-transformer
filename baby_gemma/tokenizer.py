import torch
import logging


logger = logging.getLogger(__name__)


class SimpleTokenizer:

    def __init__(self):
        self.stoi = dict()
        self.itos = dict()
        self.unknown = '<UNK>'
        logger.debug('Initialized SimpleTokenizer')

    def fit(self, text: str):
        logger.info('Constructing tokenizer vocabulary')
        unique_words = sorted(list(set(text.split(' '))))
        self.words = text.split(' ')
        self.stoi = {w: i for (i, w) in enumerate(unique_words)}
        self.itos = {i: w for (i, w) in enumerate(unique_words)}
        logger.info('Tokenizer vocabulary constructed with %d unique words', self.vocab_size)

    def encode(self, text: str):
        logger.debug('Encoding text with %d words', len(text.split(' ')))
        out = []
        for word in text.split(' '):
            if word not in self.stoi:
                logger.warning('Unknown token during encode: %s', word)
            out.append(self.stoi[word])
        encoded = torch.tensor(out, dtype=torch.long)
        logger.debug('Encoded tensor shape: %s', tuple(encoded.shape))
        return encoded

    def decode(self, tensor: torch.Tensor):
        logger.debug('Decoding tensor with shape: %s', tuple(tensor.shape) if tensor.dim() > 0 else '()')
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

        logger.error('Unsupported tensor dimension for decode: %d', token_tensor.dim())

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def __len__(self) -> int:
        return self.vocab_size