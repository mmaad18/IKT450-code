import time

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2

from assignments.convolutional_neural_networks_4.Food11Dataset import Food11Dataset
from assignments.convolutional_neural_networks_4.networks.LeNet import LeNet
from assignments.convolutional_neural_networks_4.networks.ResNet import ResNet
from utils import display_info, load_device, print_time

from utils import plot_loss


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
    test_loss, accuracy = 0.0, 0.0

    # Prevents PyTorch from calculating and storing gradients
    with torch.no_grad():
        for X, T in dataloader:
            X, T = X.to(device), T.to(device)

            Y = model(X)
            pred_class = Y.argmax(dim=1)
            test_loss += loss_fn(Y, T).item()
            accuracy += (pred_class == T).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy /= size

    return test_loss, accuracy


def main():
    display_info(4)
    start = time.perf_counter()

    device = load_device()
    print(f"Using {device} device")

    #model = LeNet().to(device)
    model = ResNet().to(device)
    print(model)

    learning_rate = 1e-3
    momentum = 0.9
    batch_size = 256
    epochs = 500
    decay = 0.0001

    print_time(start, "Loaded and compiled network")

    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop(size=96, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.25, hue=0.15),
        v2.ToTensor(),
        v2.RandomChoice([
            v2.GaussianNoise(mean=0.0, sigma=0.05),
            v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.0))
        ]),
        v2.Normalize(mean=[0.5607, 0.4520, 0.3385], std=[0.2598, 0.2625, 0.2692]),
    ])

    train_data = Food11Dataset("datasets/Food_11", "training", transform)
    eval_data = Food11Dataset("datasets/Food_11", "validation", transform)
    test_data = Food11Dataset("datasets/Food_11", "evaluation", transform)

    print_time(start, "Loaded data")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-7)

    test_losses = []
    accuracies = []

    for epoch in range(epochs):
        train_loop(train_loader, model, loss_fn, optimizer, device)
        test_loss, accuracy = test_loop(test_loader, model, loss_fn, device)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        scheduler.step(test_loss)

        learning_rate = optimizer.param_groups[0]['lr']

        print(f"-----------------| Epoch: {epoch} |-----------------")
        print(f"Learning Rate: {learning_rate}")
        print_time(start)
        print(f"Test Loss: {test_loss}")
        print(f"Accuracy: {accuracy}\n")

        if epochs > 10 and epoch % 50 == 0:
            plot_loss("Cross Entropy", test_losses, learning_rate, momentum, batch_size)
            plot_loss("Accuracy", accuracies, learning_rate, momentum, batch_size)

        if epoch % 10 == 0:
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6} MB")

        if epoch > 15 and test_loss > test_losses[0]:
            break

    print("Done!")
    plot_loss("Cross Entropy", test_losses[10:], learning_rate, momentum, batch_size)


main()
