
from pathlib import Path

import h5py
import torch


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, filename: str | Path, file_idx: list[int] | None = None):
        self.data = h5py.File(filename, 'r')

        self.file_idx = file_idx if file_idx is not None else list(range(self.data['mels'].shape[0]))

    def __len__(self) -> int:
        return len(self.file_idx)

    def __getitem__(self, index: int) -> torch.Tensor:
        item = self.file_idx[index]
        x = torch.as_tensor(self.data['mels'][item])
        x = torch.log(x + torch.finfo(x.dtype).eps)
        return x


class MelTokenDataset(MelDataset):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        seq_len = 10
        self.tokens = torch.zeros(len(self.file_idx), seq_len, dtype=torch.int64)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = super().__getitem__(index)
        y = self.tokens[index]
        z = torch.as_tensor(index)
        return x, y, z


class MelEmbeddingDataset(MelDataset):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        seq_len = 10
        d_embedding = 256
        self.embeddings = torch.zeros(len(self.file_idx), seq_len, d_embedding)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = super().__getitem__(index)
        y = self.embeddings[index]
        z = torch.as_tensor(index)

        return x, y, z
