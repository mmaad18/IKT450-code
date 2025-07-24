# pyright: reportConstantRedefinition=false
from typing import cast, Sized

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from numpy.typing import NDArray

from assignments.common.metrics import evaluate_metrics, plot_evaluation
from assignments.neural_networks_2.EcoliDataset import EcoliDataset
from assignments.neural_networks_2.EcoliNeuralNetwork import EcoliNeuralNetwork
from utils import display_info, load_device


def train_loop(
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        model: EcoliNeuralNetwork,
        loss_fn: BCEWithLogitsLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> None:
    model.train()

    for _, (X, T) in enumerate(dataloader):
        X, T = X.to(device), T.to(device)

        Y = model(X)
        loss = loss_fn(Y, T)

        # Backpropagation
        optimizer.zero_grad() # Reset gradients to prevent accumulation
        loss.backward()
        optimizer.step()


def test_loop(
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        model: EcoliNeuralNetwork,
        device: torch.device
) -> NDArray[np.float64]:
    model.eval()

    size = len(cast(Sized, dataloader.dataset))
    Y_val = np.zeros((size,))
    Y_pred = np.zeros((size,))

    b = 0  # Batch index

    with torch.no_grad():
        for X, T in dataloader:
            X, T = X.to(device), T.to(device)

            pred = model(X)
            pred_class = (pred > 0.5).float()

            Y_val[b:b + len(T)] = T.cpu().flatten().numpy()
            Y_pred[b:b + len(pred)] = pred_class.cpu().flatten().numpy()

            b += len(T)

    return evaluate_metrics(Y_val, Y_pred)


def main():
    display_info(2)

    device: torch.device = load_device()
    print(f"Using {device} device")

    model = EcoliNeuralNetwork(device)
    print(model)

    training_data = EcoliDataset("assignments/neural_networks_2/ecoli.data", training=True)
    test_data = EcoliDataset("assignments/neural_networks_2/ecoli.data", training=False)

    learning_rate = 0.01
    momentum = 0.9
    batch_size = 10
    epochs = 1000

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    evaluation: NDArray[np.float64] = np.zeros((epochs, 10))

    for e in range(epochs):
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        evaluation[e] = test_loop(test_dataloader, model, device)

        if e % 10 == 0:
            MSE= evaluation[e][8]
            print(f"Epoch {e}\n-------------------------------")
            print(f"Test MSE: {MSE}\n")


    print("Done!")
    plot_evaluation(evaluation, "Epoch", f" (η={learning_rate}, α={momentum}, batch_size={batch_size})")


main()
