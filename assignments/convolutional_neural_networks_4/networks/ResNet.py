import torch
from torch import nn, Tensor


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=1, dilation=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv1 = conv3x3(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels, in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.relu(out)

        return out


class ResizeBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        out_channels = 2 * in_channels

        self.conv1 = conv3x3(in_channels, out_channels, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            conv1x1(in_channels, out_channels, stride=2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 48x48x64
            BasicBlock(in_channels=64),
            BasicBlock(in_channels=64),
            # 24x24x128
            ResizeBlock(in_channels=64),
            BasicBlock(in_channels=128),
            # 12x12x256
            ResizeBlock(in_channels=128),
            BasicBlock(in_channels=256),
            # 6x6x512
            ResizeBlock(in_channels=256),
            BasicBlock(in_channels=512),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(512, 11),
        )

        self._initialize_weights()


    def forward(self, x):
        return self.network_stack(x)


    def _initialize_weights(self):
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)
        """

        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                torch.nn.init.zeros_(layer.bias)



