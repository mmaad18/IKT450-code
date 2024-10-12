import numpy as np
import torch
from torch.utils.data import Dataset

class EcoliDataset(Dataset):
    def __init__(self, file_path, training=False, split_ratio=0.8):
        self.file_path = file_path
        self.training = training
        self.split_ratio = split_ratio
        self.X, self.Y = self.data_preprocessing(file_path)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    def data_preprocessing(self, file_path: str):
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
        index = int(len(filtered_data) * self.split_ratio)
        if self.training:
            filtered_data = filtered_data[:index]
        else:
            filtered_data = filtered_data[index:]

        # Split into input (X) and output (Y) variables
        X = filtered_data[:, 1:8].astype(float)
        Y = filtered_data[:, 8].astype(float)

        return X, Y.reshape(-1, 1)

