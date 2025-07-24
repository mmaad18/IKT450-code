# pyright: reportConstantRedefinition=false, reportMissingTypeStubs=false
import time
from typing import cast, Sized

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from assignments.common.metrics import evaluate_metrics, plot_evaluation
from assignments.convolutional_neural_networks_4.Food11Dataset import Food11Dataset
from assignments.convolutional_neural_networks_4.networks.LeNet import LeNet
from utils import display_info, load_device, print_time


def train_loop(
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        model: nn.Module,
        loss_fn: CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> None:
    model.train()

    for _, (X, T) in enumerate(dataloader):
        X, T = X.to(device), T.to(device)

        Y = model(X)
        Y_class = Y.argmax(dim=1)
        loss = loss_fn(Y_class, T)

        # Backpropagation
        optimizer.zero_grad() # Reset gradients to prevent accumulation
        loss.backward()
        optimizer.step()


def test_loop(
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        model: nn.Module,
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
    display_info(4)
    start = time.perf_counter()

    device: torch.device = load_device()
    print(f"Using {device} device")

    model = LeNet().to(device)
    print(model)

    learning_rate = 1e-3
    momentum = 0.9
    batch_size = 256
    epochs = 50
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
    # eval_data = Food11Dataset("datasets/Food_11", "validation", transform)
    test_data = Food11Dataset("datasets/Food_11", "evaluation", transform)

    print_time(start, "Loaded data")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-7)

    evaluation: NDArray[np.float64] = np.zeros((epochs, 10))

    for e in range(epochs):
        train_loop(train_loader, model, loss_fn, optimizer, device)
        evaluation[e] = test_loop(test_loader, model, device)

        scheduler.step(evaluation[e][8])

        learning_rate = optimizer.param_groups[0]['lr']

        print(f"-----------------| Epoch: {e} |-----------------")
        print(f"Learning Rate: {learning_rate}")
        print_time(start)
        print(f"Test MSE: {evaluation[e][8]}")
        print(f"Test Accuracy: {evaluation[e][7]}\n")

        if e % 10 == 0:
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6} MB")

    print("Done!")
    plot_evaluation(evaluation, "Epoch", f" (η={learning_rate}, α={momentum}, batch_size={batch_size})")


main()
