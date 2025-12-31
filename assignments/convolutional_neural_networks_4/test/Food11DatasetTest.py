import unittest
import time

import torch
from torch.utils.data import DataLoader

from assignments.convolutional_neural_networks_4.Food11Dataset import Food11Dataset
from assignments.convolutional_neural_networks_4.util_4 import get_base_transform, get_test_transform
from utils import print_time


class MyTestCase(unittest.TestCase):
    root_path = "C:\\Users\\mmbio\\Documents\\GitHub\\IKT450-code\\datasets\\Food_11"


    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


    def test_loading_time(self):
        start = time.perf_counter()

        print_time(start, "Loading training data - START")
        train_data = Food11Dataset("datasets/Food_11", "training", get_base_transform(), get_train_transform())
        print_time(start, "Loading training data complete - COMPLETE")
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
        print_time(start, "Creating DataLoader complete - COMPLETE")


    def test_calculate_mean_and_std_dev(self):
        dataset = Food11Dataset(self.root_path, "training", get_base_transform(), get_test_transform())
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        total_sum = torch.zeros(3)
        total_squared_sum = torch.zeros(3)
        total_pixels = 0

        for images, _ in dataloader:
            batch_size, channels, height, width = images.shape
            batch_pixels = batch_size * height * width
            total_pixels += batch_pixels

            # Sum over batch and spatial dimensions (height and width)
            total_sum += images.sum(dim=[0, 2, 3])
            total_squared_sum += (images ** 2).sum(dim=[0, 2, 3])

        mean = total_sum / total_pixels
        std = torch.sqrt((total_squared_sum / total_pixels) - (mean ** 2))

        print(f"Mean: {mean}")
        print(f"Std: {std}")


if __name__ == '__main__':
    unittest.main()
