import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from DatasetMode import DatasetMode as DM


class FishImageDataset(Dataset):
    def __init__(self, root_path, mode=DM.TRAIN, split_ratio=0.8):
        self.root_path = root_path
        self.mode = mode
        self.split_ratio = split_ratio


    def __len__(self):
        return 23


    def __getitem__(self, index):
        img_path = os.path.join(self.root_path, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


    def get_file_path(self):


        return self.root_path
