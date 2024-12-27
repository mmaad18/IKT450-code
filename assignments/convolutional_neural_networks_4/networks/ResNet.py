import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x) + x


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network_stack = nn.Sequential(
            # 96x96x3
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 48x48x32
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48*48*32, 11),
        )

        self._initialize_weights()


    def forward(self, x):
        return self.network_stack(x)


    def _initialize_weights(self):
        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal(layer.weight)
                torch.nn.init.zeros_(layer.bias)


