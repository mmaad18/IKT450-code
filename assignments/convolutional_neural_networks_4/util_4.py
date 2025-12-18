import torch
from torchvision.transforms import v2


def get_train_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        v2.RandomHorizontalFlip(0.5),
        v2.ColorJitter(brightness=0.05, hue=0.03),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.5607, 0.4520, 0.3385],
            std=[0.2598, 0.2625, 0.2692],
        ),
    ])


def get_test_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((96, 96)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.5607, 0.4520, 0.3385],
            std=[0.2598, 0.2625, 0.2692],
        ),
    ])

