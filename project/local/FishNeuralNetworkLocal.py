import torch
from torch import nn


class FishNeuralNetworkLocal(nn.Module):
    def __init__(self):
        super().__init__()

        self.network_stack = nn.Sequential(
            # 64x64x3
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=7, padding=3),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
            # 32x32x4
            nn.Conv2d(in_channels=4, out_channels=5, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
            # 16x16x5
            nn.Conv2d(in_channels=5, out_channels=6, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
            # 8x8x6
            nn.Flatten(),
            nn.Linear(8*8*6, 4*4*6),
            nn.Sigmoid(),
            nn.Linear(4*4*6, 23),
        )

        self._initialize_weights()


    def forward(self, x):
        return self.network_stack(x)


    def _initialize_weights(self):
        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


