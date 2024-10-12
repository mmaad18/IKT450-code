from torch import nn


class EcoliNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network_stack = nn.Sequential(
            nn.Linear(7, 3),
            nn.Sigmoid(),
            nn.Linear(3, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network_stack(x)