import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from DatasetMode import DatasetMode as DM


class FishDataset(Dataset):
    def __init__(self, file_path, mode=DM.TRAIN, split_ratio=0.8):
        self.file_path = file_path
        self.mode = mode
        self.split_ratio = split_ratio

        self.img_labels = [
            'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit'
        ]


    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, index):
        img_path = os.path.join(self.file_path, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


