import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.device = device

        self.network_stack = nn.Sequential(
            # 96x96x3
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 48x48x96
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 24x24x256
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 12x12x384
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            # 12x12x384
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            # 12x12x384
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 6x6x256
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(6 * 6 * 256, 3072),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 11),
        ).to(self.device)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network_stack(x)


    def short_name(self) -> str:
            return "Alex"


    def _initialize_weights(self):
        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)


