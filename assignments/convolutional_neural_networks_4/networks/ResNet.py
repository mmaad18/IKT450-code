import torch
from torch import nn, Tensor


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=1, dilation=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResizeBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        out_channels = 2

        self.conv1 = conv3x3(in_channels, out_channels, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network_stack = nn.Sequential(
            # 96x96x3
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 48x48x64
            BasicBlock(in_channels=64, out_channels=64),
            BasicBlock(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2),
            # 24x24x128
            BasicBlock(in_channels=128, out_channels=128),
            BasicBlock(in_channels=128, out_channels=256),
            nn.MaxPool2d(kernel_size=2),
            # 12x12x256
            BasicBlock(in_channels=256, out_channels=256),
            BasicBlock(in_channels=256, out_channels=512),
            nn.MaxPool2d(kernel_size=2),
            # 6x6x512
            BasicBlock(in_channels=512, out_channels=512),
            BasicBlock(in_channels=512, out_channels=512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 11),
        )

        self._initialize_weights()


    def forward(self, x):
        return self.network_stack(x)


    def _initialize_weights(self):
        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                torch.nn.init.zeros_(layer.bias)


