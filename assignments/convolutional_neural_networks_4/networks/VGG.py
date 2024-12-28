import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network_stack = nn.Sequential(
            # 96x96x3
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 48x48x6
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 24x24x12
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 12x12x24
            nn.Flatten(),
            nn.Linear(12*12*24, 9*9*24),
            nn.ReLU(),
            nn.Linear(9*9*24, 6*6*24),
            nn.ReLU(),
            nn.Linear(6*6*24, 11),
        )

        self._initialize_weights()


    def forward(self, x):
        return self.network_stack(x)


    def _initialize_weights(self):
        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


