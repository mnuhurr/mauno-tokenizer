import math
import torch

from dataclasses import dataclass

from .backbone import TransformerEncoder
from .enc import ConvEncoder
from .penc import PositionalEncoding
from .utils import mask_tokens


@dataclass
class ModelConfig:
    n_mels: int
    n_enc_channels: int
    d_model: int
    n_layers: int = 12
    n_heads: int = 8
    dropout: float = 0.2
    #p_masking: float = 0.0
    masking_length: int = 50
    max_tokens: int = 1024

    n_tokens: int = 1024


class Model(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.encoder = ConvEncoder(
            n_mels=config.n_mels,
            n_channels=config.n_enc_channels,
            d_model=config.d_model,
            dropout=config.dropout)

        self.positional_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_sequence_length=config.max_tokens,
            max_random_offset=150)

        self.backbone = TransformerEncoder(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout)

        self.ln = torch.nn.LayerNorm(config.d_model)
        self.head = torch.nn.Linear(config.d_model, config.n_tokens)

        self.register_parameter('mask_token', torch.nn.Parameter(torch.randn(config.d_model) / math.sqrt(config.d_model)))

        self.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        # todo: mask stuff
        # todo: specaugment?
        x = self.encoder(x)
        if self.training:
            seq_len = x.size(1)
            max_offset = seq_len - self.config.masking_length
            starts = torch.randint(max_offset, size=(x.size(0),), device=x.device)
            masking_mask = torch.zeros(x.size(0), seq_len, device=x.device)

            for k, s in enumerate(starts):
                x[k, s:s + self.config.masking_length] = self.mask_token
                masking_mask[k, s:s + self.config.masking_length] = 1
        else:
            masking_mask = None

        x = self.positional_encoding(x)
        z = self.backbone(x)
        x = self.ln(z[-1])
        x = self.head(x)
        return x, z, masking_mask
