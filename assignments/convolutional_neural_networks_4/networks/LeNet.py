from typing import Mapping, Any

import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.device = device

        self.network_stack = nn.Sequential(
            # 96x96x3
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7, padding=3),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 48x48x6
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 24x24x12
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 12x12x24
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(12*12*24, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 11),
        ).to(self.device)

        self._initialize_weights()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network_stack(x)


    def _initialize_weights(self):
        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                torch.nn.init.zeros_(layer.bias)

    """
    def get_metadata(self) -> Mapping[str, Any]:
        return {
            "device": str(self.device),
            "input_dim": self.policy_net[0].in_features,
            "output_dim": self.policy_net[-1].out_features,
            "layer_sizes": [layer.out_features for layer in self.policy_net if isinstance(layer, nn.Linear)],
            "learning_rate": self.learning_rate,
            "memory_capacity": self.memory_capacity,
            "memory_size": len(self.memory),
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay": self.eps_decay,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
        }
        """

