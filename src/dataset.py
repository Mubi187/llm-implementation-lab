import torch
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        """
        data       : full tokenized text as a 1D tensor
        block_size : context window length
        """
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # every position up to last full window is valid
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # input: block_size tokens starting at idx
        x = self.data[idx: idx + self.block_size]
        # target: same window shifted right by 1
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y
