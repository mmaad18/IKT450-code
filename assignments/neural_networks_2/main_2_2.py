import torch
from torch import nn
from torch.utils.data import DataLoader

from assignments.neural_networks_2.EcoliDataset import EcoliDataset
from assignments.neural_networks_2.EcoliNeuralNetwork import EcoliNeuralNetwork
from assignments.neural_networks_2.main_2_utils import plot_loss
from utils import display_info, load_device


def train_loop(dataloader, model, loss_fn, optimizer, device="cpu"):
    model.train()

    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        pred = model(X)
        loss = loss_fn(pred, Y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # Reset gradients to prevent accumulation


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
            pred_class = (pred > 0.5).float()
            test_loss += loss_fn(pred, Y).item()
            correct += (pred_class == Y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return test_loss


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

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    test_losses = []

    for t in range(epochs):
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loss = test_loop(test_dataloader, model, loss_fn, device)
        test_losses.append(test_loss)

        if t % 10 == 0:
            print(f"Epoch {t}\n-------------------------------")
            print(f"Test Error: {test_loss}\n")

    print("Done!")
    plot_loss("MSE", test_losses, learning_rate, momentum, batch_size)


main()
