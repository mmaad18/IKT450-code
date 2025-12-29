# pyright: reportUnknownMemberType=false
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from torch.utils.data import Dataset, DataLoader, random_split
from collections.abc import Sized
from typing import cast, Any, Mapping, Union


def print_time(start: float, message: str="Elapsed time") -> None:
    end = time.perf_counter()
    print(f"[LOG] ---- {message}: {end - start} seconds")


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


def display_memory_usage() -> None:
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6} MB\n")


def dataset_to_loaders_3(
        dataset: Dataset[torch.Tensor],
        batch_size: int,
        train_factor: float=0.7,
        val_factor: float=0.2,
        num_workers: int=0
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    train_size = int(train_factor * len(cast(Sized, dataset)))
    val_size = int(val_factor * len(cast(Sized, dataset)))
    test_size = len(cast(Sized, dataset)) - train_size - val_size
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
    train_size = int(train_factor * len(cast(Sized, dataset)))
    val_size = len(cast(Sized, dataset)) - train_size
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


def plot_list(list_values: list[float], title: str="Title") -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(list_values, label=f"{title}", color='blue', linewidth=2)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel(f"{title}", fontsize=16)
    plt.title(f"{title} vs Epochs", fontsize=20)

    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()


def create_run_id(network_name: str) -> str:
    run_time = datetime.fromtimestamp(time.time()).strftime("%y%m%d_%H%M%S")
    return network_name + "_" + run_time


def logs_path(run_id: str, base_path: str = "output/logs") -> Path:
    run_folder = Path(base_path) / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def get_state_dict(obj: Union[Optimizer, LRScheduler]) -> dict:
    return {
        "type": obj.__class__.__name__,
        "state_dict": obj.state_dict(),
    }


def save_metadata(run_id: str, batch_size: int, epochs: int, optimizer: Optimizer, scheduler: LRScheduler) -> None:
    metadata_path = logs_path(run_id) / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": get_state_dict(optimizer),
        "scheduler": get_state_dict(scheduler),
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to: {metadata_path}")


def save_commentary(run_id: str, network_structure: str) -> None:
    comment_path = logs_path(run_id) / "comment.md"
    comment_path.parent.mkdir(parents=True, exist_ok=True)

    run_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S")

    comment = f"""
# Comments

### Run ID
{run_id}

### Time of Run
{run_time}

### Network Structure
{network_structure}

    """

    with open(comment_path, 'w') as f:
        f.write(comment)

    print(f"Comment saved to: {comment_path}")


def load_metrics(run_id: str) -> dict[str, np.ndarray]:
    file_path = logs_path(run_id) / "metrics.npz"
    with np.load(file_path) as loaded:
        print(f"Loading metrics from: {file_path}")
        return dict(loaded)


def save_model(run_id: str, model: torch.nn.Module) -> None:
    file_path = logs_path(run_id) / "model.pth"
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to: {file_path}")


def load_model(run_id: str, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    file_path = logs_path(run_id) / "model.pth"
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"Model loaded from: {file_path}")
    return model

