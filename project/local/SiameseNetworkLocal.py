import torch.nn as nn
import torch.nn.functional as F

from project.local.FishNeuralNetworkLocal import FishNeuralNetworkLocal


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size=128):
        super(SiameseNetwork, self).__init__()

        # Shared subnetwork
        self.backbone = FishNeuralNetworkLocal()

    def forward(self, input1, input2):
        # Get embeddings for both inputs
        embedding1 = self.backbone(input1)
        embedding2 = self.backbone(input2)

        # Calculate the distance (e.g., L2 Norm)
        distance = F.pairwise_distance(embedding1, embedding2, p=2)  # Euclidean distance

        return distance
