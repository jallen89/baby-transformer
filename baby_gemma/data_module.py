from dataclasses import dataclass
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset


class TextChunkDataset(Dataset):

    def __init__(self, tokens, seq_len):

        self.tokens = tokens 
        self.seq_len = seq_len

    def __len__(self):
        # Get amount of tokens - 1 (there is no label for this token) / seq_len
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        # Get next token as the label for all samples. 
        y = self.tokens[start + 1:end + 1]
        return x, y 


@dataclass
class DataModuleConfig:
    seq_len: int = 64
    batch_size: int = 8
    split_ratios: Tuple[float, float, float] = (0.80, 0.10, 0.10)  # train, val, test
    pin_memory: bool = True
    num_workers: int = 0

class TextDataModule:

    def __init__(self, tokens, cfg=DataModuleConfig()):
        self.cfg = cfg 
        self._setup(tokens)

    def _setup(self, tokens):

        train_ratio = self.cfg.split_ratios[0]
        val_ratio = self.cfg.split_ratios[1]

        n = len(tokens)
        n_train = int(train_ratio*n)
        n_val = int(val_ratio*n)

        train_tokens = tokens[:n_train]
        val_tokens = tokens[n_train: n_train + n_val]
        test_tokens = tokens[n_train + n_val:]

        self.train_dataset = TextChunkDataset(train_tokens, self.cfg.seq_len)
        self.val_dataset = TextChunkDataset(val_tokens, self.cfg.seq_len)
        self.test_dataset = TextChunkDataset(test_tokens, self.cfg.seq_len)

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_dataset)

    def val_dataloader(self) -> DataLoader: 
        return self._create_dataloader(self.val_dataset)

    def _create_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=self.cfg.pin_memory,
            num_workers=self.cfg.num_workers
        )