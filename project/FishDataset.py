import os

from torch.utils.data import Dataset
from torchvision.io import read_image

from project.FishRecord import FishRecord


class FishDataset(Dataset):
    def __init__(self, root_path, prefix):
        self.root_path = root_path
        self.prefix = prefix
        self.data_list = self.label_processing()


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        X = read_image(self.data_list[index].file_path)
        Y = self.data_list[index].species

        return X, Y


    def label_processing(self):
        label_file_path = os.path.join(self.root_path, "class_id.csv")

        with open(label_file_path, 'r') as file:
            next(file)
            data_list = [FishRecord(self.root_path, self.prefix, line.strip().split()[0]) for line in file]

        return data_list

