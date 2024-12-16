import os

import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class Food11Dataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Compose):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        self.labels = [
            'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit'
        ]

        self.data_list = self.label_processing()
        self.X, self.T = self.data_preprocessing()


    def __len__(self):
        return len(self.T)


    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]


    def label_processing(self):
        data_list = []

        for label in self.labels:
            label_dir = os.path.join(self.root_dir, label)
            label_idx = self.labels.index(label)

            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                data_list.append((file_path, label_idx))

        return data_list


    def data_preprocessing(self):
        X_list = []
        T_list = []

        for file_path, label_idx in self.data_list:
            image_X = Image.open(file_path)
            tensor_X = self.transform(image_X)
            X_list.append(tensor_X)
            T_list.append(label_idx)

        return X_list, T_list


