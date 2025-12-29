import unittest

import torch

from assignments.convolutional_neural_networks_4.networks.VggNet import VggNet
from utils import load_device, get_state_dict


class MyTestCase(unittest.TestCase):
    def test_model_print(self):
        device: torch.device = load_device()
        print(f"Using {device} device")

        model = VggNet(device)
        print(model)


    def test_optimizer_print(self):
        device: torch.device = load_device()
        model = VggNet(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            threshold=1e-2,
            min_lr=1e-6
        )
        print(get_state_dict(optimizer))
        print(get_state_dict(scheduler))


if __name__ == '__main__':
    unittest.main()
