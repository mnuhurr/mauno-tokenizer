import torch


"""
transformer

"""


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        self.ln0 = torch.nn.LayerNorm(d_model)
        self.mha = torch.nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.ln1 = torch.nn.LayerNorm(d_model)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ln0(x)
        x_attn, scores = self.mha(x, x, x)
        x = x + x_attn

        x = self.ln1(x)
        x = x + self.fc(x)

        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> list[torch.Tensor]:
        y = []

        for layer in self.layers:
            x = layer(x)
            y.append(x)

        return y


"""
conformer

"""


class Swish(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()


class FFN(torch.nn.Sequential):
    def __init__(self, d_model: int, d_fc: int | None = None, dropout: float = 0.0):
        d_fc = d_fc if d_fc is not None else 4 * d_model
        super().__init__(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_fc),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_fc, d_model))


class ConvBlock(torch.nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 15, exp_factor: int = 4, dropout: float = 0.0):
        super().__init__()

        d_hid = exp_factor * d_model
        self.norm = torch.nn.LayerNorm(d_model)

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(d_model, 2 * d_hid, kernel_size=1),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(d_hid, d_hid, kernel_size=kernel_size, groups=d_hid, padding='same'),
            torch.nn.BatchNorm1d(d_hid),
            Swish(),
            torch.nn.Conv1d(d_hid, d_model, kernel_size=1),
            torch.nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        return x


class ConformerBlock(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int = 4,
                 kernel_size: int = 31,
                 exp_factor: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.fc1 = FFN(d_model, dropout=dropout)
        self.mha = torch.nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.conv = ConvBlock(d_model=d_model, kernel_size=kernel_size, exp_factor=exp_factor, dropout=dropout)
        self.fc2 = FFN(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.fc1(x)
        attn_out, scores = self.mha(x, x, x, key_padding_mask=mask)
        x = x + attn_out
        x = x + self.conv(x)
        x = x + self.fc2(x)

        return x


class Conformer(torch.nn.Sequential):
    def __init__(self,
                 d_model: int,
                 n_blocks: int,
                 n_heads: int = 4,
                 kernel_size: int = 31,
                 exp_factor: int = 4,
                 dropout: float = 0.1):
        super().__init__(*[
            ConformerBlock(d_model=d_model,
                           n_heads=n_heads,
                           kernel_size=kernel_size,
                           exp_factor=exp_factor,
                           dropout=dropout) for _ in range(n_blocks)
        ])

