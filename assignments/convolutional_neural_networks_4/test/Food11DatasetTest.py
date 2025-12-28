import unittest
import time

from torch.utils.data import DataLoader

from assignments.convolutional_neural_networks_4.Food11Dataset import Food11Dataset
from assignments.convolutional_neural_networks_4.util_4 import get_train_transform, get_base_transform
from utils import print_time


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


    def test_loading_time(self):
        start = time.perf_counter()

        print_time(start, "Loading training data - START")
        train_data = Food11Dataset("datasets/Food_11", "training", get_base_transform(), get_train_transform())
        print_time(start, "Loading training data complete - COMPLETE")
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
        print_time(start, "Creating DataLoader complete - COMPLETE")


if __name__ == '__main__':
    unittest.main()
