import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from project.FishRecord import FishRecord


class FishDataset(Dataset):
    def __init__(self, root_path, prefix, transform):
        self.root_path = root_path
        self.prefix = prefix
        self.transform = transform
        self.data_list = self.label_processing()


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        X = Image.open(self.data_list[index].file_path)
        X = self.transform(X)
        Y = self.data_list[index].species

        to_pil = transforms.ToPILImage()
        pil_image = to_pil(X)
        pil_image.save(f"out/output_image_{index}.png")

        return X, Y


    def label_processing(self):
        label_file_path = os.path.join(self.root_path, "class_id.csv")

        with open(label_file_path, 'r') as file:
            next(file)
            data_list = [FishRecord(self.root_path, self.prefix, line.strip().split()[0]) for line in file]

        return data_list

