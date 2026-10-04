import pytest
from ..tokenizer import SimpleTokenizer
from ..tokenizer import BPETokenizer
import torch

@pytest.fixture
def tokenizer():
    t = SimpleTokenizer()
    dataset = "A cat in the hat"
    t.fit(dataset)
    return t

@pytest.fixture
def bpe_tokenizer():
    t = BPETokenizer(27)
    dataset = "the cat is in the tree with the frogs and the dog. They have a cow. banana"
    t.fit(dataset)
    return t 

def fit(tokenizer: SimpleTokenizer):
    assert (
        tokenizer.stoi['A'] == 0 
        and tokenizer.stoi['cat'] == 1
        and tokenizer.stoi['hat'] == 2
        and tokenizer.stoi['in'] == 3
        and tokenizer.stoi['the'] == 4
    )

def test_encode(tokenizer: SimpleTokenizer):
    print(tokenizer.stoi)
    tokens = tokenizer.encode("cat hat")
    assert torch.equal(tokens, torch.tensor([1, 2], dtype=torch.float))

def test_decode_single(tokenizer: SimpleTokenizer):
    decoded = tokenizer.decode(torch.tensor([1, 2], dtype=torch.float))
    assert decoded == "cat hat"

def test_decode_batch(tokenizer: SimpleTokenizer):
    decoded = tokenizer.decode(
        torch.tensor(
            [
                [1, 2],
                [3, 4],
            ],
            dtype=torch.float,
        )
    )
    assert decoded == ["cat hat", "in the"]



def test_bpe_decoder(bpe_tokenizer: BPETokenizer):
    encoded = bpe_tokenizer.encode('cat tree banana a $')
    decoded = bpe_tokenizer.decode(encoded)
    assert decoded == f"cat tree banana a {bpe_tokenizer.unknown}"


def test_bpe_e2e():
    corpus = "the cat is in the tree with the frogs and the dog. They have a cow. banana"
    tok = BPETokenizer(k_vocab=40)
    # Ensure base class attributes are initialized if not handled by super().__init__()
    if not hasattr(tok, 'special_tokens'):
        tok.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        tok.unknown = '<unk>'
        tok.stoi = {}
        tok.itos = {}

    tok.fit(corpus)

    test_sentence = "the cat in the tree"
    encoded = tok.encode(test_sentence)
    decoded = tok.decode(encoded)

    print(f"Vocab size: {len(tok.stoi)}")
    print(f"Encoded IDs: {encoded}")
    print(f"Decoded: '{decoded}'")
    assert decoded == test_sentence, f"Mismatch: '{decoded}' != '{test_sentence}'"

# encoded = tokenizer.encode(sentence)
# decoded = tokenizer.decode(encoded)
# print("sentence", sentence)
# print("encoder:", encoded)
# print("decoded:", decoded)


