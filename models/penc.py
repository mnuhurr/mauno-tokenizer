import math
import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_sequence_length: int = 1024, max_random_offset: int = 0):
        super().__init__()

        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_sequence_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.max_random_offset = max_random_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.max_random_offset > 0:
            offset = torch.randint(self.max_random_offset, size=())
        else:
            offset = 0
        x = x + self.pe[:, offset:offset + x.size(1)]
        return x

