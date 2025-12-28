import torch
from torchvision.transforms import v2


def get_base_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((112, 112)),
        v2.ToDtype(torch.float32, scale=True),
    ])


def get_train_transform() -> v2.Compose:
    return v2.Compose([
        v2.RandomCrop((96, 96)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
        v2.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
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

