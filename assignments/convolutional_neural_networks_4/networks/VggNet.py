from typing import Mapping, Any

import torch
from torch import nn, Tensor
from torchvision.models.resnet import conv3x3


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.network_stack = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network_stack(x)


class VggNet(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        self.network_stack = nn.Sequential(
            # 96x96x3
            BasicBlock(in_channels=3, out_channels=64),
            # 48x48x64
            BasicBlock(in_channels=64, out_channels=128),
            # 24x24x128
            BasicBlock(in_channels=128, out_channels=256),
            # 12x12x256
            BasicBlock(in_channels=256, out_channels=512),
            # 6x6x512
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 11),
        ).to(self.device)

        self._initialize_weights()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network_stack(x)


    def short_name(self) -> str:
        return "Vgg"


    def _initialize_weights(self):
        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)


