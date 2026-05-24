class CharTokenizer:
    def __init__(self, text: str):
        # all unique characters in the dataset = vocabulary
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        # mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return ''.join([self.itos[i] for i in tokens])

    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size})"
