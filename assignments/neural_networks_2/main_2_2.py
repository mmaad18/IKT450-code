import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from assignments.neural_networks_2.NeuralNetwork import NeuralNetwork
from assignments.neural_networks_2.main_2_utils import data_preprocessing
from assignments.utils import display_info


def main():
    display_info(2)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    X_train, X_val, Y_train, Y_val, filtered_data = data_preprocessing("assignments/neural_networks_2/ecoli.data")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

    X = torch.rand(1, 8, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")


main()
