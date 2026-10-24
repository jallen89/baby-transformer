import pytest
from ..tokenizer import SimpleTokenizer
from ..tokenizer import BPETokenizer
from ..tokenizer import BPETokenizerHF
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


def test_hf_bpe_token_cache_round_trip(tmp_path):
    corpus = "the cat sat on the mat " * 20
    cache_prefix = tmp_path / "corpus-bpe"

    first_tokenizer = BPETokenizerHF(k_vocab=300)
    first_tokens = first_tokenizer.fit_encode_cached(corpus, cache_prefix)

    tokenizer_path = tmp_path / "corpus-bpe.tokenizer.json"
    tokens_path = tmp_path / "corpus-bpe.tokens.pt"
    assert tokenizer_path.exists()
    assert tokens_path.exists()

    second_tokenizer = BPETokenizerHF(k_vocab=300)
    second_tokens = second_tokenizer.fit_encode_cached(corpus, cache_prefix)

    assert torch.equal(first_tokens, second_tokens)
    assert second_tokenizer.decode(second_tokens) == corpus


def test_hf_bpe_token_cache_invalidates_for_new_text(tmp_path):
    cache_prefix = tmp_path / "corpus-bpe"
    tokenizer = BPETokenizerHF(k_vocab=300)
    original = tokenizer.fit_encode_cached("alpha beta " * 20, cache_prefix)

    tokenizer = BPETokenizerHF(k_vocab=300)
    changed = tokenizer.fit_encode_cached("completely different text " * 20, cache_prefix)

    assert not torch.equal(original, changed)

# encoded = tokenizer.encode(sentence)
# decoded = tokenizer.decode(encoded)
# print("sentence", sentence)
# print("encoder:", encoded)
# print("decoded:", decoded)


