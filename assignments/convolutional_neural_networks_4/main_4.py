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

from assignments.common.ConfusionMatrix import ConfusionMatrix
from assignments.common.metrics import plot_cross_entropy
from assignments.convolutional_neural_networks_4.Food11Dataset import Food11Dataset
from assignments.convolutional_neural_networks_4.networks.AlexNet import AlexNet
from assignments.convolutional_neural_networks_4.networks.LeNet import LeNet
from assignments.convolutional_neural_networks_4.networks.VggNet import VggNet
from assignments.convolutional_neural_networks_4.util_4 import get_train_transform, get_test_transform
from utils import display_info, load_device, print_time, plot_list


def train_loop(
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        model: nn.Module,
        loss_fn: CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> float:
    model.train()
    loss_sum = 0.0
    n_total = 0

    for X, T in dataloader:
        X, T = X.to(device), T.to(device).long()

        Y = model(X)
        loss = loss_fn(Y, T)

        # Backpropagation
        optimizer.zero_grad() # Reset gradients to prevent accumulation
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * T.size(0)
        n_total += T.size(0)

    return loss_sum / n_total


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

    avg_ce = ce_sum / n_total
    return Y_val, Y_pred, avg_ce


def main():
    display_info(4)
    start = time.perf_counter()

    device: torch.device = load_device()
    print(f"Using {device} device")

    model = VggNet(device)
    print(model)

    print_time(start, "Loaded and compiled network")

    train_data = Food11Dataset("datasets/Food_11", "training", get_train_transform())
    val_data = Food11Dataset("datasets/Food_11", "validation", get_test_transform())
    test_data = Food11Dataset("datasets/Food_11", "evaluation", get_test_transform())

    print_time(start, "Loaded datasets")

    batch_size = 256

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print_time(start, "Created data loaders")

    learning_rate = 3e-4
    momentum = 0.9
    weight_decay = 1e-3
    epochs = 200

    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs // 4, eta_min=1e-6)

    print_time(start, "Created optimizer and scheduler")

    # METRICS
    avg_ces: NDArray[np.float64] = np.zeros(epochs)
    train_ces: NDArray[np.float64] = np.zeros(epochs)
    lr_list: list[float] = []

    confusion_matrix = ConfusionMatrix(len(train_data.labels), train_data.short_labels)
    confusion_matrix_aggregate = ConfusionMatrix(len(train_data.labels), train_data.short_labels)
    confusion_matrix_metrics = np.zeros((epochs, confusion_matrix.metrics_size))
    confusion_matrix_aggregate_metrics = np.zeros((epochs, confusion_matrix_aggregate.metrics_size))

    print_time(start, "Starting training")

    for e in range(epochs):
        train_ce = train_loop(train_loader, model, loss_fn, optimizer, device)
        Y_val, Y_pred, avg_ce = test_loop(val_loader, model, loss_fn, device)

        # METRICS
        confusion_matrix.update_vector(Y_val, Y_pred, True)
        confusion_matrix_aggregate.update_vector(Y_val, Y_pred, False)
        confusion_matrix_metrics[e] = confusion_matrix.metrics()
        confusion_matrix_aggregate_metrics[e] = confusion_matrix_aggregate.metrics()

        scheduler.step(avg_ce)  # pyright: ignore[reportUnknownMemberType]
        #scheduler.step()
        learning_rate = optimizer.param_groups[0]['lr']

        avg_ces[e] = avg_ce
        train_ces[e] = train_ce
        lr_list.append(learning_rate)

        print(f"-----------------| Epoch: {e} |-----------------")
        print_time(start)
        print(f"Learning Rate: {learning_rate}")
        print(f"Test Cross-Entropy: {avg_ce}\n")

        if e % (epochs // 10) == 0 and e > 0:
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6} MB\n")


    print_time(start, "Training complete")

    confusion_matrix_test = ConfusionMatrix(len(test_data.labels), test_data.short_labels)
    Y_val, Y_pred, avg_ce = test_loop(test_loader, model, loss_fn, device)
    confusion_matrix_test.update_vector(Y_val, Y_pred, True)
    print("Average test Cross-Entropy:", avg_ce)

    print_time(start, "Test complete")

    # DONE
    plot_list(lr_list, "Learning Rate")
    plot_cross_entropy(train_ces, "Epoch", ", Training")
    plot_cross_entropy(avg_ces, "Epoch", f" (η={learning_rate}, α={momentum}, b={batch_size})")
    confusion_matrix.plotly_plot_metrics(confusion_matrix_metrics)
    confusion_matrix.plotly_plot()
    confusion_matrix_aggregate.plotly_plot_metrics(confusion_matrix_aggregate_metrics)
    confusion_matrix_aggregate.plotly_plot()
    confusion_matrix_test.plotly_plot()

    print_time(start, "Plots generated")


main()

