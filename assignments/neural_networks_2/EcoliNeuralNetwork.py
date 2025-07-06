import torch
from torch import nn


class EcoliNeuralNetwork(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.device = device

        self.network_stack = nn.Sequential(
            nn.Linear(7, 3),
            nn.Sigmoid(),
            nn.Linear(3, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        ).to(self.device)

        self._initialize_weights()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network_stack(x)


    def _initialize_weights(self) -> None:
        for layer in self.network_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


