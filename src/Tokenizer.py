

class CharTokenizer:
    def __init__(self):
        self.char_to_index = {}
        self.index_to_char = {}
        self.vocab_size = 0

    def fit(self, text):
        unique_chars = sorted(set(text))
        self.vocab_size = len(unique_chars)
        self.char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}

    def encode(self, text):
        return [self.char_to_index[char] for char in text]
    
    def decode(self, indices):
        return ''.join([self.index_to_char[idx] for idx in indices])