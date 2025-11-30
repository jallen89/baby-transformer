import pytest
import torch
from .tokenizer import Tokenizer

def test_vocab_size():
    t = Tokenizer(lang='en')
    sentence = 'What are you doing?'
    t.add_words_from_sentence(sentence)
    print(t.vocab_size)
    assert t.vocab_size == len(sentence.split(' ')) + len(t.special_chars)

def test_add_sos_eos():
    t = Tokenizer(lang='en')
    sentence = 'What are you doing?'
    t.add_words_from_sentence(sentence)
    encoded_sentence = t.encode_sentence(sentence, add_sos_eos=True)
    
    sos_token_id = t.word_to_id['<sos>']
    eos_token_id = t.word_to_id['<eos>']
    
    assert encoded_sentence[0] == sos_token_id
    assert encoded_sentence[-1] == eos_token_id
    assert len(encoded_sentence) == len(sentence.split()) + 2

def test_encode_sentence():
    t = Tokenizer(lang='en')
    sentence = "a test sentence"
    t.add_words_from_sentence(sentence)
    encoded = t.encode_sentence(sentence)
    
    expected_ids = [
        t.word_to_id['a'],
        t.word_to_id['test'],
        t.word_to_id['sentence']
    ]
    
    assert torch.equal(encoded, torch.tensor(expected_ids, dtype=torch.long))

def test_encode_unknown_word():
    t = Tokenizer(lang='en')
    sentence = "a test"
    t.add_words_from_sentence(sentence)
    encoded = t.encode_sentence("a test with unknown word")
    
    unk_token_id = t.word_to_id['<UNK>']
    
    expected_ids = [
        t.word_to_id['a'],
        t.word_to_id['test'],
        unk_token_id,
        unk_token_id,
        unk_token_id,
    ]
    
    assert torch.equal(encoded, torch.tensor(expected_ids, dtype=torch.long))

def test_decode_sentence():
    t = Tokenizer(lang='en')
    sentence = "a test Sentence"
    t.add_words_from_sentence(sentence)
    
    encoded = t.encode_sentence(sentence)
    decoded = t.decode(encoded)
    
    assert decoded == sentence.lower()

def test_decode_with_special_tokens():
    t = Tokenizer(lang='en')
    sentence = "a test seNtenCe"
    t.add_words_from_sentence(sentence)
    
    encoded = t.encode_sentence(sentence, add_sos_eos=True)
    decoded = t.decode(encoded)
    
    expected_sentence = f"<sos> {sentence.lower()} <eos>"
    assert decoded == expected_sentence

def test_decode_unknown_token():
    t = Tokenizer(lang='en')
    sentence = "a test"
    t.add_words_from_sentence(sentence)
    
    # Create a list of token IDs including one that doesn't exist in the vocab
    known_ids = t.encode_sentence("a test").tolist()
    unknown_id = 999 # An ID that is not in the vocabulary
    token_ids_with_unknown = known_ids + [unknown_id]
    
    decoded = t.decode(token_ids_with_unknown)
    assert decoded == "a test <UNK>"


def test_save_and_load_vocab():
    # 1. Create and train an initial tokenizer
    t1 = Tokenizer(lang='en')
    sentence = "this is a test sentence for saving and loading"
    t1.add_words_from_sentence(sentence)

    # 2. Save the vocabulary to a temporary file
    vocab_file = '/tmp/vocab.json'
    t1.save_vocab(vocab_file)

    # 3. Load the vocabulary into a new tokenizer instance
    t2 = Tokenizer.load_vocab(vocab_file)

    # 4. Assert that the loaded tokenizer has the same state
    assert t1.lang == t2.lang
    assert t1.word_to_id == t2.word_to_id
    assert t1.id_to_word == t2.id_to_word
    assert t1.vocab_size == t2.vocab_size
    assert t1._next_id == t2._next_id

    # 5. Assert that the loaded tokenizer behaves identically
    test_sentence = "a test for loading"
    encoded_by_t1 = t1.encode_sentence(test_sentence)
    encoded_by_t2 = t2.encode_sentence(test_sentence)
    assert torch.equal(encoded_by_t1, encoded_by_t2)

    decoded_by_t2 = t2.decode(encoded_by_t1)
    assert decoded_by_t2 == test_sentence.lower()
    