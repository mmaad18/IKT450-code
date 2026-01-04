import torch
from torch import nn, Tensor
from torchvision.models.resnet import conv1x1, conv3x3


class DenseLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.network_stack = nn.Sequential(
            conv1x1(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            conv3x3(channels, channels),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.network_stack(x) + x)


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


class DenseNet(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.network_stack = nn.Sequential(
            # 96x96x3
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 48x48x64
            DenseLayer(64),
            DenseLayer(64),
            DenseLayer(64),
            # 24x24x128
            ResizeBlock(in_channels=64, out_channels=128),
            DenseLayer(128),
            DenseLayer(128),
            DenseLayer(128),
            # 12x12x256
            ResizeBlock(in_channels=128, out_channels=256),
            DenseLayer(256),
            DenseLayer(256),
            DenseLayer(256),
            DenseLayer(256),
            DenseLayer(256),
            # 6x6x512
            ResizeBlock(in_channels=256, out_channels=512),
            DenseLayer(512),
            DenseLayer(512),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 11),
        ).to(self.device)

        self._initialize_weights()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network_stack(x)


    def short_name(self) -> str:
        return "Dense"


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




