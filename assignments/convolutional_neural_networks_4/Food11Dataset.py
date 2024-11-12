import os

from PIL import Image
from torch.utils.data import Dataset


class Food11Dataset(Dataset):
    def __init__(self, root_path, prefix, transform):
        self.root_path = root_path
        self.prefix = prefix
        self.transform = transform

        self.data_list = [
            'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit'
        ]
        self.X, self.T = self.data_preprocessing()


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        return self.X[index], self.T[index]


    def data_preprocessing(self):
        X_list = []
        T_list = []

        for record in self.data_list:
            file_path = os.path.join(self.root_path, record)
            image_X = Image.open(record.file_path)
            tensor_X = self.transform(image_X)
            X_list.append(tensor_X)
            T_list.append(record.species)

        return X_list, T_list


