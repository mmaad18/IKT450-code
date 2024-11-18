import os

from PIL import Image
from torch.utils.data import Dataset

from project.FishRecord import FishRecord


class FishDatasetRemote(Dataset):
    def __init__(self, root_path, prefix, transform, device="cpu"):
        self.root_path = root_path
        self.prefix = prefix
        self.transform = transform
        self.device = device
        self.data_list = self.label_processing()
        self.X, self.T = self.data_preprocessing()


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]


    def label_processing(self):
        label_file_path = os.path.join(self.root_path, "class_id.csv")

        with open(label_file_path, 'r') as file:
            next(file)
            data_list = [FishRecord(self.root_path, self.prefix, line.strip().split()[0]) for line in file]

        return data_list


    def data_preprocessing(self):
        X_list = []
        T_list = []

        for record in self.data_list:
            image_X = Image.open(record.file_path)
            tensor_X = self.transform(image_X)
            X_list.append(tensor_X.to(self.device))
            T_list.append(record.species.to(self.device))

        return X_list, T_list