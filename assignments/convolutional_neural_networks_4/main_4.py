# pyright: reportConstantRedefinition=false, reportMissingTypeStubs=false
import time
from typing import cast, Sized

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import CrossEntropyLoss
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from assignments.common.ConfusionMatrix import ConfusionMatrix
from assignments.common.metrics import plot_cross_entropy
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
        X, T = X.to(device), T.to(device).long()

        Y = model(X)
        loss = loss_fn(Y, T)

        # Backpropagation
        optimizer.zero_grad() # Reset gradients to prevent accumulation
        loss.backward()
        optimizer.step()


def test_loop(
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        model: nn.Module,
        loss_fn: CrossEntropyLoss,
        device: torch.device
) -> tuple[NDArray[np.int32], NDArray[np.int32], float]:
    model.eval()

    size = len(cast(Sized, dataloader.dataset))
    Y_val = np.zeros((size,), dtype=np.int32)
    Y_pred = np.zeros((size,), dtype=np.int32)

    b = 0  # Batch index
    ce_sum = 0.0
    n_total = 0

    with torch.no_grad():
        for X, T in dataloader:
            X, T = X.to(device), T.to(device).long()
            logits  = model(X)
            pred = logits .argmax(dim=1)

            ce = loss_fn(logits, T)
            ce_sum += float(ce.item()) * T.size(0)
            n_total += T.size(0)

            n = T.shape[0]
            Y_val[b:b+n] = T.cpu().flatten().numpy()
            Y_pred[b:b+n] = pred.cpu().flatten().numpy()
            b += n

    avg_ce = ce_sum / max(1, n_total)
    return Y_val, Y_pred, avg_ce


def main():
    display_info(4)
    start = time.perf_counter()

    device: torch.device = load_device()
    print(f"Using {device} device")

    model = LeNet(device)
    print(model)

    learning_rate = 0.1
    momentum = 0.9
    batch_size = 256
    epochs = 200
    decay = 0.0001

    print_time(start, "Loaded and compiled network")

    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop(size=96, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        #v2.ColorJitter(brightness=0.25, hue=0.15),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        #v2.RandomChoice([
        #    v2.GaussianNoise(mean=0.0, sigma=0.05),
        #    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.0))
        #]),
        #v2.Normalize(mean=[0.5607, 0.4520, 0.3385], std=[0.2598, 0.2625, 0.2692]),
    ])

    train_data = Food11Dataset("datasets/Food_11", "training", transform)
    # eval_data = Food11Dataset("datasets/Food_11", "validation", transform)
    test_data = Food11Dataset("datasets/Food_11", "evaluation", transform)

    print_time(start, "Loaded data")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.95, eps=1e-06, weight_decay=decay)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-7)

    avg_ces: NDArray[np.float64] = np.zeros(epochs)

    confusion_matrix = ConfusionMatrix(len(train_data.labels), train_data.short_labels)
    confusion_matrix_aggregate = ConfusionMatrix(len(train_data.labels), train_data.short_labels)

    confusion_matrix_metrics = np.zeros((epochs, confusion_matrix.metrics_size))
    confusion_matrix_aggregate_metrics = np.zeros((epochs, confusion_matrix_aggregate.metrics_size))

    for e in range(epochs):
        train_loop(train_loader, model, loss_fn, optimizer, device)
        Y_val, Y_pred, avg_ce = test_loop(test_loader, model, loss_fn, device)
        confusion_matrix.update_vector(Y_val, Y_pred, True)
        confusion_matrix_aggregate.update_vector(Y_val, Y_pred, False)

        confusion_matrix_metrics[e] = confusion_matrix.metrics()
        confusion_matrix_aggregate_metrics[e] = confusion_matrix_aggregate.metrics()

        # scheduler.step(evaluation[e][8])  # pyright: ignore[reportUnknownMemberType]
        # learning_rate = optimizer.param_groups[0]['lr']

        avg_ces[e] = avg_ce

        print(f"-----------------| Epoch: {e} |-----------------")
        print(f"Learning Rate: {learning_rate}")
        print_time(start)
        print(f"Test Cross-Entropy: {avg_ce}\n")

        if e % (epochs // 10) == 0 and e > 0:
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6} MB")

            confusion_matrix.plotly_plot_metrics(confusion_matrix_metrics)
            confusion_matrix.plotly_plot(f" (epoch={e})")

            learning_rate = 0.9 * learning_rate


    print("Done!")

    plot_cross_entropy(avg_ces, "Epoch", f" (η={learning_rate}, α={momentum}, b={batch_size})")
    confusion_matrix.plotly_plot_metrics(confusion_matrix_metrics)
    confusion_matrix.plotly_plot()
    confusion_matrix_aggregate.plotly_plot_metrics(confusion_matrix_aggregate_metrics)
    confusion_matrix_aggregate.plotly_plot()


main()

