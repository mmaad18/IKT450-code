import torch
from torch import nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, dilation=dilation, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential(nn.Identity())

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network_stack = nn.Sequential(
            # 96x96x3
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 48x48x32
            BasicBlock(in_channels=32, out_channels=32),
            nn.ReLU(),
            BasicBlock(in_channels=32, out_channels=32),
            nn.ReLU(),
            BasicBlock(in_channels=32, out_channels=32),
            nn.ReLU(),
            BasicBlock(in_channels=32, out_channels=32),
            nn.ReLU(),
            BasicBlock(in_channels=32, out_channels=32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(24*24*32, 1000),
            nn.Softmax(dim=1),
        )

        self._initialize_weights()


    def forward(self, x):
        return self.network_stack(x)


    def _initialize_weights(self):
        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                torch.nn.init.zeros_(layer.bias)


