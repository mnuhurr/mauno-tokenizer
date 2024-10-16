import torch


class ConvEncoder(torch.nn.Module):
    def __init__(self, n_mels: int, d_model: int, n_channels: int, dropout: float = 0.0):
        super().__init__()

        self.cnn = torch.nn.Sequential(
            #torch.nn.Conv1d(n_mels, n_channels, kernel_size=5, stride=5),
            torch.nn.Conv1d(n_mels, n_channels, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm1d(n_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            #torch.nn.Conv1d(n_channels, n_channels, kernel_size=5, stride=2, padding=2),
            torch.nn.Conv1d(n_channels, n_channels, kernel_size=5, stride=5),
            torch.nn.BatchNorm1d(n_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv1d(n_channels, d_model, kernel_size=3, padding='same')
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        return x




def _test():
    x = torch.randn(4, 128, 3000)
    enc = ConvEncoder(n_mels=128, d_model=256, n_channels=512, dropout=0.2)
    y = enc(x)
    print(y.shape)


if __name__ == '__main__':
    _test()
