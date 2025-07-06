import time
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split


def print_time(start: float, message: str="Elapsed time") -> None:
    end = time.perf_counter()
    print(f"{message}: {end - start} seconds")


def display_info(assignment_number: int) -> None:
    print("Author: Mohamed Yahya Maad")
    print("Course: IKT450")
    print(f"Assignment: {assignment_number}")


def display_info_project() -> None:
    print("Author: Mohamed Yahya Maad")
    print("Course: IKT450")
    print("Project: Fish Classification")
    print("CUDA Available: " + str(torch.cuda.is_available()))
    print("GPU Name: " + str(torch.cuda.get_device_name(0)))


def load_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )


def dataset_to_loaders_3(
        dataset: Dataset[torch.Tensor],
        batch_size: int,
        train_factor: float=0.7,
        val_factor: float=0.2,
        num_workers: int=0
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    train_size = int(train_factor * len(dataset))
    val_size = int(val_factor * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, eval_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    train_loader: DataLoader[torch.Tensor] = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader: DataLoader[torch.Tensor] = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader: DataLoader[torch.Tensor] = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, eval_loader, test_loader


def dataset_to_loaders_2(
        dataset: Dataset[torch.Tensor],
        batch_size: int,
        train_factor: float=0.8,
        num_workers: int=0
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    train_size = int(train_factor * len(dataset))
    val_size = len(dataset) - train_size
    train_data, eval_data = random_split(dataset, [train_size, val_size])

    train_loader: DataLoader[torch.Tensor] = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader: DataLoader[torch.Tensor] = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, eval_loader


def plot_loss(loss_type: str, losses: list[float], eta: float, alpha: float, batch_size: int) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label=f"{loss_type} over epochs", color='blue', linewidth=2)

    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel(f"{loss_type}", fontsize=16)
    plt.title(f"{loss_type} vs Epochs (eta={eta}, alpha={alpha}, batch_size={batch_size})", fontsize=20)

    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()

