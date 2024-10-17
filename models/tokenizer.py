import math
import torch

from dataclasses import dataclass
from vector_quantize_pytorch import VectorQuantize

from .backbone import TransformerEncoder
from .enc import ConvEncoder
from .penc import PositionalEncoding
from .utils import mask_tokens


@dataclass
class TokenizerConfig:
    n_mels: int
    n_enc_channels: int
    d_model: int
    n_layers: int = 12
    n_heads: int = 8
    dropout: float = 0.2
    p_masking: float = 0.0
    max_tokens: int = 1024

    codebook_size: int = 1024
    codebook_dim: int = 64


class Tokenizer(torch.nn.Module):
    def __init__(self, config: TokenizerConfig):
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

        self.proj0 = torch.nn.Linear(config.d_model, config.codebook_dim)

        self.quantizer = VectorQuantize(
            dim=config.codebook_dim,
            codebook_size=config.codebook_size,
            use_cosine_sim=True,
            threshold_ema_dead_code=2)

        self.proj1 = torch.nn.Sequential(
            torch.nn.Linear(config.codebook_dim, config.d_model),
            TransformerEncoder(d_model=config.d_model, n_layers=3, n_heads=config.n_heads, dropout=config.dropout))

        self.register_parameter('mask_token', torch.nn.Parameter(torch.randn(config.d_model) / math.sqrt(config.d_model)))
        self.commit_loss = None

        self.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, project: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        # todo: mask stuff
        # todo: specaugment?
        x = self.encoder(x)
        if self.training and self.config.p_masking > 0:
            mask_tokens(x, self.mask_token, p=self.config.p_masking)

        x = self.positional_encoding(x)
        x = self.backbone(x)[-1]
        x = torch.tanh(x)
        x = self.proj0(x)
        x, codes, self.commit_loss = self.quantizer(x)
        if project:
            x = self.proj1(x)[-1]
        return codes, x
