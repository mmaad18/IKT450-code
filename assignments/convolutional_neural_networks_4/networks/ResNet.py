import torch
from torch import nn, Tensor
from torchvision.models.resnet import conv1x1, conv3x3


class BasicBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.network_stack = nn.Sequential(
            conv3x3(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            conv3x3(channels, channels),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.network_stack(x) + x)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1):
        super().__init__()
        out_channels = 4 * mid_channels

        self.network_stack = nn.Sequential(
            conv1x1(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            conv3x3(mid_channels, mid_channels, stride=stride),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            conv1x1(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.network_stack(x) + self.shortcut(x))


class ResizeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.network_stack = nn.Sequential(
            conv3x3(in_channels, out_channels, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential(
            conv1x1(in_channels, out_channels, stride=2),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.network_stack(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, device: torch.device, network_stack: nn.Sequential) -> None:
        super().__init__()
        self.device = device
        self.network_stack = network_stack.to(self.device)
        self._initialize_weights()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network_stack(x)


    def _short_name(self, layers: int) -> str:
        return "Res" + str(layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


