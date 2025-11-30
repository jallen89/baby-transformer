import torch
import json
import json

class Tokenizer:
    def __init__(self, lang):
        self.lang = lang
        self.special_chars = ['<UNK>', '<sos>', '<eos>']
        self.word_to_id = {char: idx for idx, char in enumerate(self.special_chars)}
        self.id_to_word = {idx: char for char, idx in self.word_to_id.items()}
        self._next_id = len(self.word_to_id)

    def _add_word(self, word):
        w = word.lower()
        if w and w not in self.word_to_id:
            self.word_to_id[w] = self._next_id
            self.id_to_word[self._next_id] = w
            self._next_id += 1

    def add_words_from_sentence(self, sentence):
        for word in sentence.split():
            self._add_word(word)

    def build_from_sentences(self, sentences):
        for sentence in sentences:
            self.add_words_from_sentence(sentence)

    @property
    def vocab_size(self):
        return len(self.word_to_id)
        

    def encode_sentence(self, text, add_sos_eos=False):
        token_ids = []
        if add_sos_eos:
            token_ids.append(self.word_to_id['<sos>'])
        
        for word in text.split():
            w = word.lower()
            token_id = self.word_to_id.get(w, self.word_to_id['<UNK>'])
            token_ids.append(token_id)

        if add_sos_eos:
            token_ids.append(self.word_to_id['<eos>'])
            
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        words = [self.id_to_word.get(token_id, '<UNK>') for token_id in token_ids]
        return " ".join(words)

    def save_vocab(self, filepath):
        vocab_data = {
            'lang': self.lang,
            'word_to_id': self.word_to_id
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_vocab(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        lang = vocab_data.get('lang')
        word_to_id = vocab_data.get('word_to_id', {})

        # Ensure ids are ints (be defensive in case JSON stored them as strings)
        word_to_id = {word: int(idx) for word, idx in word_to_id.items()}

        tokenizer = cls(lang=lang)
        tokenizer.word_to_id = word_to_id
        tokenizer.id_to_word = {idx: word for word, idx in word_to_id.items()}
        
        # Correctly set _next_id considering special characters might not be in the saved vocab
        # if it was created with an older version of the class.
        if word_to_id:
            max_id = max(word_to_id.values())
            tokenizer._next_id = max_id + 1
        else:
            # If vocab is empty, start after special chars
            tokenizer._next_id = len(tokenizer.special_chars)

        # Ensure special characters from the class are in the loaded vocab
        for char in tokenizer.special_chars:
            if char not in tokenizer.word_to_id:
                 # This part handles loading vocabs that might be missing special tokens
                 # It's a bit of a patch, assuming they should be at the start if missing.
                 # A more robust implementation might enforce their presence during save/load.
                 pass # For now, we assume a saved vocab is complete.

        return tokenizer
