import torch
import logging
from collections import Counter


logger = logging.getLogger(__name__)


class Tokenizer:

    def __init__(self):
        self.stoi = dict()
        self.itos = dict()
        self.unknown = '<UNK>'
        self.special_tokens = [
            self.unknown,
        ]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def __len__(self) -> int:
        return self.vocab_size
class SimpleTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
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



class BPETokenizer(Tokenizer):

    def __init__(self, k_vocab):
        super().__init__()
        logger.debug('Initialized SimpleTokenizer')
        self.k_vocab = k_vocab 
        self.vocab = set()
        self.merges = dict()


    def _sentence_to_words(self, sentence):
        return [' ' + w for w in sentence.split(' ') if w]


    def fit(self, text: str):
        logger.info('Constructing tokenizer vocabulary')

        words = self._sentence_to_words(text)
        for word in words:
            self.vocab = self.vocab.union(set(word))

        # Count words in dict to get initial weights. 
        counts = Counter(words)
        char_counts = Counter({tuple(k): v for (k,v) in counts.items()})

        num_merges = self.k_vocab - len(self.vocab) - len(self.special_tokens)

        for rank in range(max(0, num_merges)):

            # Calculate the pair frequencies over each char count. 
            pair_frequences = Counter()
            for key, weight in char_counts.items():
                n = len(key)
                if n >= 2:
                    for i in range(n - 1):
                        pair_frequences[(key[i], key[i + 1])] += weight


            # There are no merges remaining. 
            if len(pair_frequences) == 0:
                break 

            # Determine the most common pair.
            most_common = pair_frequences.most_common()[0][0]
            most_common_key = ''.join(most_common)
            self.vocab.add(most_common_key)

            self.merges[most_common] = rank

            # If pair sequence matches the most_common pair then replace it. 
            new_char_counts = Counter()
            for key, weight in char_counts.items():
                n = len(key)
                new_key = list()
                i = 0 
                while i < n:
                    if i == n - 1 or most_common != tuple([key[i], key[i + 1]]):
                        new_key.append(key[i])
                        i += 1
                    else:
                        new_key.append(''.join(key[i:i+2]))
                        i += 2

                new_char_counts[tuple(new_key)] += weight

            char_counts = new_char_counts

        # Create vocab ids
        tokens = self.special_tokens + list(sorted(self.vocab))
        for idx, c in enumerate(tokens):
            self.stoi[c] = idx
            self.itos[idx] = c

    def encode(self, sentence):

        words = self._sentence_to_words(sentence)

        output = []
        for word in words:
            tokens = self._encode_word(word)
            for token in tokens:
                id = self.stoi[token]
                output.append(id)

        return torch.tensor(output, dtype=torch.long)
        
    def _encode_word(self, word):

        # Split input chars. 
        tokens = [char if char in self.vocab else self.unknown for char in word]
        n = len(tokens)

        while len(tokens) >= 2:

            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            n = len(tokens)

            min_rank, min_pair = float('inf'), None
            # Get lowest ordered rank. 

            for pair in pairs:
                rank = self.merges.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank 
                    min_pair = pair

            if min_pair is None:
                break

            # Update tokens to replace lowest-ranked ordered pair. 
            new_tokens = []
            i = 0
            while i <= n - 1:
                if i != n - 1 and (tokens[i], tokens[i + 1]) == min_pair:
                    new_tokens.append(''.join(min_pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def decode(self, tokens):
        text = ''.join(self.itos[token.item()] for token in tokens)
        return text.lstrip(' ')
