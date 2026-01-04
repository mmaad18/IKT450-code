import torch
from torch import nn

from assignments.convolutional_neural_networks_4.networks.ResNet import BasicBlock, ResizeBlock, ResNet


class ResNet34(ResNet):
    def __init__(self, device: torch.device) -> None:
        network_stack = nn.Sequential(
            # 96x96x3
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 48x48x64
            BasicBlock(64),
            BasicBlock(64),
            BasicBlock(64),
            # 24x24x128
            ResizeBlock(in_channels=64, out_channels=128),
            BasicBlock(128),
            BasicBlock(128),
            BasicBlock(128),
            # 12x12x256
            ResizeBlock(in_channels=128, out_channels=256),
            BasicBlock(256),
            BasicBlock(256),
            BasicBlock(256),
            BasicBlock(256),
            BasicBlock(256),
            # 6x6x512
            ResizeBlock(in_channels=256, out_channels=512),
            BasicBlock(512),
            BasicBlock(512),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(512, 11),
        )

        super().__init__(device, network_stack)

    def short_name(self) -> str:
        return self._short_name(34)


