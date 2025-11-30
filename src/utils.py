import logging
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .tokenizer import Tokenizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)


class FrenchEnglishDataset(Dataset):

    def __init__(self, filename):
        self.filename = filename
        self.english_tokenizer = Tokenizer(lang="english")
        self.french_tokenizer = Tokenizer(lang="french")

    def __len__(self):
        return self.english_tensor.size(0)

    def __getitem__(self, idx):
        return {
            "english": self.english_tensor[idx],
            "french": self.french_tensor[idx],
        }

    def intialize_dataset(self):

        # Get translation pairs. 
        translation_pairs = self.load_data(self.filename)

        # Build tokenizers. 
        self._build_tokenizers(translation_pairs)

        # encode and pad translation pairs
        self._encode_and_pad(translation_pairs)

    def _encode_and_pad(self, translation_pairs):
        """Encodes and pads the translation pairs."""

        english_sequences = []
        french_sequences = []

        for english_sentence, french_sentence in translation_pairs:
            # Convert English sentence to tensor
            encoded_english_sentence = self.english_tokenizer.encode_sentence(english_sentence)
            english_sequences.append(encoded_english_sentence)
            
            # Convert French sentence to tensor, adding <sos> and <eos> tokens
            encoded_french_sentence = self.french_tokenizer.encode_sentence(french_sentence, add_sos_eos=True)
            french_sequences.append(encoded_french_sentence)

        # Pad sequences using PyTorch's pad_sequence utility
        self.english_tensor = pad_sequence(english_sequences, batch_first=True, padding_value=0)
        self.french_tensor = pad_sequence(french_sequences, batch_first=True, padding_value=0)

    def load_data(self, filename):
        """Loads translation pairs from a file."""
        logger.info("Loading data from %s", filename)
        translation_pairs = []
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    english = parts[0].strip()
                    french = parts[1].strip()
                    translation_pairs.append((english, french))

        logger.info("Loaded %d translation pairs", len(translation_pairs))
        return translation_pairs


    def _build_tokenizers(self, translation_pairs):
        """Builds tokenizers from the translation pairs."""
        english_sentences = [en for en, _ in translation_pairs]
        french_sentences = [fr for _, fr in translation_pairs]
        self.english_tokenizer.build_from_sentences(english_sentences)
        self.french_tokenizer.build_from_sentences(french_sentences)
        logger.info("Built English vocab size=%d French vocab size=%d",
                    self.english_tokenizer.vocab_size, self.french_tokenizer.vocab_size)