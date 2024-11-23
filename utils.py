import time
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split


def time_function(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"{func.__name__} took {end - start} seconds")
    return result


def time_function_out(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    diff = end - start
    print(f"{func.__name__} took {diff} seconds")
    return result, diff


def display_info(assignment_number: int):
    print("Author: Mohamed Yahya Maad")
    print("Course: IKT450")
    print(f"Assignment: {assignment_number}")


def display_info_project():
    print("Author: Mohamed Yahya Maad")
    print("Course: IKT450")
    print("Project: Fish Classification")


def load_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def dataset_to_loaders(dataset, batch_size: int, train_factor=0.7, val_factor=0.2, num_workers=0):
    train_size = int(train_factor * len(dataset))
    val_size = int(val_factor * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, eval_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, eval_loader, test_loader


def plot_loss(loss_type: str, losses, eta: float, alpha: float, batch_size: int):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label=f"{loss_type} over epochs", color='blue', linewidth=2)

    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel(f"{loss_type}", fontsize=16)
    plt.title(f"{loss_type} vs Epochs (eta={eta}, alpha={alpha}, batch_size={batch_size})", fontsize=20)

    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()

