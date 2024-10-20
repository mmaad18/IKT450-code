import os
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split


def dataset_to_loaders(dataset, batch_size: int, train_factor=0.7, val_factor=0.2):
    train_size = int(train_factor * len(dataset))
    val_size = int(val_factor * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, eval_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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


def images_size(root_path: str):
    sizes = []
    paths = []

    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".png"):
                file_path = os.path.join(folder_path, file)
                with Image.open(file_path) as img:
                    sizes.append(img.size)
                    paths.append(file_path)

    return np.array(sizes), paths


def path_to_fish_id(path: str):
    file_name = path.split('\\')[-1]
    return int(file_name.split('_')[-1].split('.')[0])


