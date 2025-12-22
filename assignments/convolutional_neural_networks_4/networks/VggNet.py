import torch
from torch import nn, Tensor


def Conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.network_stack = nn.Sequential(
            Conv3x3(in_channels=in_channels, out_channels=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Conv3x3(in_channels=out_channels, out_channels=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x) -> torch.Tensor:
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
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(6*6*512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 11),
        ).to(self.device)

        self._initialize_weights()


    def forward(self, x):
        return self.network_stack(x)


    def _initialize_weights(self):
        # Using self.modules() ensures we find layers even if they are nested
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


