import pytest
from ..tokenizer import SimpleTokenizer
import torch

@pytest.fixture
def tokenizer():
    t = SimpleTokenizer()
    dataset = "A cat in the hat"
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
