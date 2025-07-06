import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

class EcoliDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, file_path: str, training: bool=False, split_ratio: float=0.8) -> None:
        self.file_path = file_path
        self.training = training
        self.split_ratio = split_ratio
        self.X, self.Y = self.data_preprocessing(file_path)


    def __len__(self) -> int:
        return len(self.Y)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.X[idx]
        Y = self.Y[idx]

        return torch.tensor(X, dtype=torch.float64), torch.tensor(Y, dtype=torch.float64)


    def data_preprocessing(self, file_path: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        with open(file_path, 'r') as file:
            data_list = [line.strip().split() for line in file]

        # Filter out 'cp' and 'im' labels
        data = np.array(data_list)
        mask = (data[:, -1] == "cp") | (data[:, -1] == "im")
        filtered_data = data[mask]

        # Replace 'cp' with '1' and 'im' with '0' in the last column
        labels = filtered_data[:, -1]
        filtered_data[:, -1] = np.where(labels == "cp", 1.0, 0.0)

        # Select training or validation data
        idx = int(len(filtered_data) * self.split_ratio)
        if self.training:
            filtered_data = filtered_data[:idx]
        else:
            filtered_data = filtered_data[idx:]

        # Split into input (X) and output (Y) variables
        X = filtered_data[:, 1:8].astype(np.float64)
        Y = filtered_data[:, 8].astype(np.float64)

        return X, Y.reshape(-1, 1)

