import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from assignments.neural_networks_2.EcoliDataset import EcoliDataset
from assignments.neural_networks_2.EcoliNeuralNetwork import EcoliNeuralNetwork
from assignments.neural_networks_2.main_2_utils import data_preprocessing
from assignments.utils import display_info


def load_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device="cpu"):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        pred = model(X)
        loss = loss_fn(pred, Y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # Reset gradients to prevent accumulation

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device="cpu"):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Prevents PyTorch from calculating and storing gradients
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    device = load_device()
    print(f"Using {device} device")

    model = EcoliNeuralNetwork().to(device)
    print(model)

    training_data = EcoliDataset("assignments/neural_networks_2/ecoli.data", training=True)
    test_data = EcoliDataset("assignments/neural_networks_2/ecoli.data", training=False)

    learning_rate = 1e-3
    batch_size = 10
    epochs = 5

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")


main()
