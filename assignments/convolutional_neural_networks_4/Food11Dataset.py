import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose  # pyright: ignore [reportMissingTypeStubs]


class Food11Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root_dir: str, split: str, transform: Compose):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        self.labels = [
            'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit'
        ]

        self.data_list = self.label_processing()
        self.X, self.T = self.data_preprocessing()


    def __len__(self) -> int:
        return len(self.T)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.T[idx]


    def label_processing(self) -> list[tuple[str, int]]:
        data_list: list[tuple[str, int]] = []

        for label in self.labels:
            label_dir = os.path.join(self.root_dir, label)
            label_idx = self.labels.index(label)

            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                data_list.append((file_path, label_idx))

        return data_list


    def data_preprocessing(self) -> tuple[torch.Tensor, torch.Tensor]:
        X_list: list[torch.Tensor] = []
        T_list: list[int] = []

        for file_path, idx in self.data_list:
            image_X = Image.open(file_path)
            tensor_X: torch.Tensor = self.transform(image_X)
            X_list.append(tensor_X)
            T_list.append(idx)

        X = torch.stack(X_list)  # shape: (N, C, H, W)
        T = torch.tensor(T_list, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)

        return X, T



