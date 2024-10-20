import torch
from torch import nn


class FishNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network_stack = nn.Sequential(
            # 256x256x3
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=9, padding=4),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
            # 128x128x6
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=7, padding=3),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
            # 64x64x12
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
            # 32x32x24
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
            # 16x16x48
            nn.Flatten(),
            nn.Linear(16*16*48, 8*8*48),
            nn.Sigmoid(),
            nn.Linear(8*8*48, 4*4*48),
            nn.Sigmoid(),
            nn.Linear(4*4*48, 2*2*48),
            nn.Sigmoid(),
            nn.Linear(2*2*48, 48),
            nn.Sigmoid(),
            nn.Linear(48, 23),
            nn.Softmax()
        )

        self._initialize_weights()


    def forward(self, x):
        return self.network_stack(x)


    def _initialize_weights(self):
        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


