import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from assignments.convolutional_neural_networks_4.Food11Dataset import Food11Dataset
from assignments.convolutional_neural_networks_4.Food11NeuralNetwork import Food11NeuralNetwork
from utils import display_info, load_device

from project.main_project_utils import plot_loss


def train_loop(dataloader, model, loss_fn, optimizer, device="cpu"):
    model.train()

    for batch, (X, T) in enumerate(dataloader):
        X, T = X.to(device), T.to(device)

        Y = model(X)
        loss = loss_fn(Y, T)

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
        for X, T in dataloader:
            X, T = X.to(device), T.to(device)

            Y = model(X)
            pred_class = Y.argmax(dim=0)
            test_loss += loss_fn(Y, T).item()
            correct += (pred_class == T).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return test_loss


def main():
    display_info(4)

    device = load_device()
    print(f"Using {device} device")

    model = Food11NeuralNetwork().to(device)
    print(model)

    transform = transforms.Compose([
        transforms.Resize(32),  # Resize the shorter side to 256 and keep the aspect ratio
        transforms.CenterCrop(32),
        transforms.ToTensor()  # Convert the image to a tensor
    ])

    train_data = Food11Dataset("datasets/Food_11", "training", transform)
    eval_data = Food11Dataset("datasets/Food_11", "evaluation", transform)
    test_data = Food11Dataset("datasets/Food_11", "testing", transform)

    batch_size = 100
    learning_rate = 0.1
    momentum = 0.9
    epochs = 100

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    start = time.perf_counter()

    test_losses = []

    for t in range(epochs):
        train_loop(train_loader, model, loss_fn, optimizer, device)
        test_loss = test_loop(test_loader, model, loss_fn, device)
        test_losses.append(test_loss)

        if t % 1 == 0:
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6} MB")

            print(f"Epoch {t}\n-------------------------------")
            print(f"Test Error: {test_loss}\n")

            end = time.perf_counter()
            print(f"Elapsed time: {end - start} seconds")

    print("Done!")
    plot_loss("MSE", test_losses, learning_rate, momentum, batch_size)


main()
