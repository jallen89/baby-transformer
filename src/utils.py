import logging
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)


class FrenchEnglishDataset(Dataset):

    def __init__(self, filename):
        self.filename = filename
        self.english_tokenizer = Tokenizer(lang="english")
        self.french_tokenizer = Tokenizer(lang="french")
        
        try:
            translation_pairs = self._read_translation_pairs()
            if not translation_pairs:
                self.english_tensor = torch.empty(0, dtype=torch.long)
                self.french_tensor = torch.empty(0, dtype=torch.long)
                return

            self._build_tokenizers(translation_pairs)
            self.english_tensor, self.french_tensor = self._encode_and_pad(translation_pairs)

        except FileNotFoundError:
            logger.exception("File not found: %s", self.filename)
            self.english_tensor = torch.empty(0, dtype=torch.long)
            self.french_tensor = torch.empty(0, dtype=torch.long)
        except Exception:
            logger.exception("Error loading data from %s", self.filename)
            self.english_tensor = torch.empty(0, dtype=torch.long)
            self.french_tensor = torch.empty(0, dtype=torch.long)

    def __len__(self):
        return self.english_tensor.size(0)

    def __getitem__(self, idx):
        return {
            "english": self.english_tensor[idx],
            "french": self.french_tensor[idx],
        }

    def _read_translation_pairs(self):
        """Reads translation pairs from the file."""
        translation_pairs = []
        logger.info("Loading translation pairs from %s", self.filename)
        with open(self.filename, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    logger.debug("Skipping empty line %d", lineno)
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    english, french = parts[0].strip(), parts[1].strip()
                    translation_pairs.append((english, french))
                    logger.debug("Loaded pair from line %d: %r -> %r", lineno, english, french)
                else:
                    logger.warning("Malformed line %d in %s: %r", lineno, self.filename, line)
        logger.info("Loaded %d translation pairs from %s", len(translation_pairs), self.filename)
        return translation_pairs

    def _build_tokenizers(self, translation_pairs):
        """Builds tokenizers from the translation pairs."""
        english_sentences = [en for en, _ in translation_pairs]
        french_sentences = [fr for _, fr in translation_pairs]
        self.english_tokenizer.build_from_sentences(english_sentences)
        self.french_tokenizer.build_from_sentences(french_sentences)
        logger.info("Built English vocab size=%d French vocab size=%d",
                    self.english_tokenizer.vocab_size, self.french_tokenizer.vocab_size)

    def _encode_and_pad(self, translation_pairs):
        """Encodes sentences into padded tensors."""
        english_sequences = []
        french_sequences = []
        # Use a common padding ID, for example from the English tokenizer
        pad_id = self.english_tokenizer.word_to_id['<UNK>']

        for english_sentence, french_sentence in translation_pairs:
            en_ids = self.english_tokenizer.encode(english_sentence, add_sos_eos=False)
            english_sequences.append(torch.tensor(en_ids, dtype=torch.long))

            fr_ids = self.french_tokenizer.encode(french_sentence, add_sos_eos=True)
            french_sequences.append(torch.tensor(fr_ids, dtype=torch.long))

        english_tensor = pad_sequence(english_sequences, batch_first=True, padding_value=pad_id)
        french_tensor = pad_sequence(french_sequences, batch_first=True, padding_value=pad_id)

        logger.info("Created English tensor with shape: %s", english_tensor.shape)
        logger.info("Created French tensor with shape: %s", french_tensor.shape)
        return english_tensor, french_tensor


class Tokenizer:

    def __init__(self, lang=None):
        self.lang = lang
        self.word_to_id = {'<UNK>': 0, '<sos>': 1, '<eos>': 2}
        self.id_to_word = {0: '<UNK>', 1: '<sos>', 2: '<eos>'}
        self._next_id = 3

    def _add_word(self, word):
        w = word.lower()
        if w and w not in self.word_to_id:
            self.word_to_id[w] = self._next_id
            self.id_to_word[self._next_id] = w
            self._next_id += 1

    def build_from_sentences(self, sentences):
        """Builds vocabulary from an iterable of sentence strings."""
        for sent in sentences:
            if not sent:
                continue
            for w in sent.split():
                self._add_word(w.strip())
        return self.vocab_size

    def encode(self, sentence, add_sos_eos=True):
        """Converts a sentence to a list of token ids."""
        ids = []
        if add_sos_eos:
            ids.append(self.word_to_id['<sos>'])
        
        words = [w.strip() for w in (sentence or "").split()]
        for w in words:
            if w:
                ids.append(self.word_to_id.get(w.lower(), self.word_to_id['<UNK>']))

        if add_sos_eos:
            ids.append(self.word_to_id['<eos>'])
        return ids

    def decode(self, ids, strip_special=True):
        """Converts a list of ids back to a sentence string."""
        words = []
        for i in ids:
            token = self.id_to_word.get(i, '<UNK>')
            if strip_special and token in ('<sos>', '<eos>'):
                continue
            words.append(token)
        return ' '.join(words)

    @property
    def vocab_size(self):
        return len(self.word_to_id)
