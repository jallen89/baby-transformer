from .utils import FrenchEnglishDataset
import pytest
import torch

def test_load_valid_pairs(tmp_path):
    p = tmp_path / "pairs.txt"
    p.write_text("Hello\tBonjour\nGoodbye\tAu revoir\n", encoding="utf-8")
    ds = FrenchEnglishDataset(str(p))
    assert ds.load_data() == [("Hello", "Bonjour"), ("Goodbye", "Au revoir")]


def test_skip_empty_and_malformed_lines(tmp_path):
    p = tmp_path / "pairs2.txt"
    p.write_text("Hi\tSalut\n\nMalformed line\nExtra\tCols\tIgnored\n  \t  \n", encoding="utf-8")
    ds = FrenchEnglishDataset(str(p))
    assert ds.load_data() == [("Hi", "Salut"), ("Extra", "Cols")]


def test_file_not_found_returns_empty():
    ds = FrenchEnglishDataset("no_such_file_xyz.txt")
    assert ds.load_data() == []
    def test_tokenizer_build_and_vocab_size():
        ds = FrenchEnglishDataset("no_such_file_for_tokenizer.txt")
        t = ds.english_tokenizer
        # build vocab from sentences (case-insensitive, unique words)
        returned = t.build_from_sentences(["Hello world", "Hello"])
        # special tokens: <UNK>, <sos>, <eos> = 3, plus 'hello' and 'world' = 2 -> total 5
        assert returned == t.vocab_size == 5
        assert 'hello' in t.word_to_id and 'world' in t.word_to_id


def test_encode_decode_and_options():
    ds = FrenchEnglishDataset("no_file_needed.txt")
    t = ds.english_tokenizer
    t.build_from_sentences(["Fast small"])
    # check ids for words
    id_fast = t.word_to_id['fast']
    id_small = t.word_to_id['small']

    # default add_sos_eos True
    encoded = t.encode("Fast small")
    assert encoded == [t.word_to_id['<sos>'], id_fast, id_small, t.word_to_id['<eos>']]

    # without sos/eos
    encoded_no_special = t.encode("Fast small", add_sos_eos=False)
    assert encoded_no_special == [id_fast, id_small]

    # decode should remove special tokens by default and return lowercase words
    decoded = t.decode(encoded)
    assert decoded == "fast small"


def test_unknown_words_map_to_unk_and_decode_shows_unk():
    ds = FrenchEnglishDataset("no_file_tokenizer.txt")
    t = ds.english_tokenizer
    t.build_from_sentences(["Known"])
    # encode an unknown word
    enc = t.encode("CompletelyUnknown")
    # should be <sos>, <UNK>, <eos>
    assert enc[0] == t.word_to_id['<sos>']
    assert enc[1] == t.word_to_id['<UNK>']
    assert enc[-1] == t.word_to_id['<eos>']
    # decoding should return the '<UNK>' token (special tokens stripped)
    assert t.decode(enc) == "<UNK>"


def test_dataset_builds_tokenizers_from_file_and_handles_spaces(tmp_path):
    p = tmp_path / "pairs_vocab.txt"
    p.write_text("One two\tUn deux\nThree\tTrois\nExtra   Spaces\tEspace   Extra\n", encoding="utf-8")
    ds = FrenchEnglishDataset(str(p))
    pairs = ds.load_data()
    assert pairs == [("One two", "Un deux"), ("Three", "Trois"), ("Extra   Spaces", "Espace   Extra")]

    # english vocab: one, two, three, extra, spaces -> 5 + 3 specials = 8
    eng_vocab = ds.english_tokenizer.vocab_size
    assert eng_vocab == 8

    # french vocab: un, deux, trois, espace, extra -> 5 + 3 specials = 8
    fr_vocab = ds.french_tokenizer.vocab_size
    assert fr_vocab == 8

    # encoding multi-word sentence (preserve order, lowercased)
    ids = ds.english_tokenizer.encode("One two", add_sos_eos=False)
    assert ids == [ds.english_tokenizer.word_to_id['one'], ds.english_tokenizer.word_to_id['two']]
    def test_read_valid_pairs(tmp_path):
        p = tmp_path / "pairs.txt"
        p.write_text("Hello\tBonjour\nGoodbye\tAu revoir\n", encoding="utf-8")
        ds = FrenchEnglishDataset(str(p))
        # Test the internal pair reading method
        assert ds._read_translation_pairs() == [("Hello", "Bonjour"), ("Goodbye", "Au revoir")]


    def test_read_skip_empty_and_malformed_lines(tmp_path):
        p = tmp_path / "pairs2.txt"
        p.write_text("Hi\tSalut\n\nMalformed line\nExtra\tCols\tIgnored\n  \t  \n", encoding="utf-8")
        ds = FrenchEnglishDataset(str(p))
        assert ds._read_translation_pairs() == [("Hi", "Salut"), ("Extra", "Cols")]


    def test_load_data_file_not_found_returns_empty_tensors():
        ds = FrenchEnglishDataset("no_such_file_xyz.txt")
        en_tensor, fr_tensor = ds.load_data()
        assert en_tensor.shape == (0,)
        assert fr_tensor.shape == (0,)


    def test_load_data_empty_file_returns_empty_tensors(tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        ds = FrenchEnglishDataset(str(p))
        en_tensor, fr_tensor = ds.load_data()
        assert en_tensor.shape == (0,)
        assert fr_tensor.shape == (0,)


    def test_load_data_returns_padded_tensors(tmp_path):
        p = tmp_path / "pairs_for_tensor.txt"
        p.write_text("One two\tUn deux\nThree\tTrois\n", encoding="utf-8")
        ds = FrenchEnglishDataset(str(p))
        en_tensor, fr_tensor = ds.load_data()

        # Check shapes: 2 sentences, max length of 2 for English, max length of 3 for French (SOS/EOS)
        assert en_tensor.shape == (2, 2)
        assert fr_tensor.shape == (2, 3)

        # Check English tensor content (no SOS/EOS, padded with <UNK>)
        en_tok = ds.english_tokenizer
        unk_id = en_tok.word_to_id['<UNK>']
        expected_en = torch.tensor([
            [en_tok.word_to_id['one'], en_tok.word_to_id['two']],
            [en_tok.word_to_id['three'], unk_id]
        ])
        assert torch.equal(en_tensor, expected_en)

        # Check French tensor content (with SOS/EOS, padded with <UNK>)
        fr_tok = ds.french_tokenizer
        expected_fr = torch.tensor([
            [fr_tok.word_to_id['<sos>'], fr_tok.word_to_id['un'], fr_tok.word_to_id['deux']],
            [fr_tok.word_to_id['<sos>'], fr_tok.word_to_id['trois'], fr_tok.word_to_id['<eos>']]
        ])
        # The padding for the second sentence in French should be <eos>, not <unk>
        # Let's re-verify the logic. The max length is 3.
        # "Un deux" -> [<sos>, un, deux] -> needs <eos> -> [<sos>, un, deux, <eos>] -> len 4
        # "Trois" -> [<sos>, trois, <eos>] -> len 3
        # Max length is 4.
        # So fr_tensor shape should be (2, 4)
        assert fr_tensor.shape == (2, 4)
        
        # Re-calculate expected French tensor with correct padding
        expected_fr_corrected = torch.tensor([
            [fr_tok.word_to_id['<sos>'], fr_tok.word_to_id['un'], fr_tok.word_to_id['deux'], fr_tok.word_to_id['<eos>']],
            [fr_tok.word_to_id['<sos>'], fr_tok.word_to_id['trois'], fr_tok.word_to_id['<eos>'], unk_id]
        ])
        assert torch.equal(fr_tensor, expected_fr_corrected)