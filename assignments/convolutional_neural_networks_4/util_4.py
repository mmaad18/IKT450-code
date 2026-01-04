import numpy as np
import torch
from torchvision.transforms import v2

from assignments.common.ConfusionMatrix import ConfusionMatrix
from utils import logs_path


base_image_size = (144, 144)
image_size = (128, 128)


def get_base_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(base_image_size),
        v2.ToDtype(torch.float32, scale=True),
    ])


def get_stats_transform() -> v2.Compose:
    return v2.Compose([
        v2.Resize(image_size),
    ])


def get_train_transform() -> v2.Compose:
    return v2.Compose([
        v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
        v2.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        v2.Normalize(
            mean=[0.5548, 0.4508, 0.3435],
            std=[0.2651, 0.2674, 0.2747],
        ),
    ])


def get_test_transform() -> v2.Compose:
    return v2.Compose([
        v2.Resize(image_size),
        v2.Normalize(
            mean=[0.5548, 0.4508, 0.3435],
            std=[0.2651, 0.2674, 0.2747],
        ),
    ])


def save_metrics(
        run_id: str,
        lr_list: list[float],
        train_ces: list[float],
        avg_ces: list[float],
        confusion_matrix_metrics: np.ndarray,
        confusion_matrix_aggregate_metrics: np.ndarray,
        confusion_matrix: ConfusionMatrix,
        confusion_matrix_aggregate: ConfusionMatrix,
        confusion_matrix_test: ConfusionMatrix
):
    file_path = logs_path(run_id) / f"metrics.npz"

    np.savez(
        file_path,
        lr_list=np.asarray(lr_list, dtype=np.float64),
        train_ces=np.asarray(train_ces, dtype=np.float64),
        avg_ces=np.asarray(avg_ces, dtype=np.float64),
        confusion_matrix_metrics=confusion_matrix_metrics,
        confusion_matrix_aggregate_metrics=confusion_matrix_aggregate_metrics,
        confusion_matrix_val=confusion_matrix.matrix,
        confusion_matrix_val_agg=confusion_matrix_aggregate.matrix,
        confusion_matrix_test=confusion_matrix_test.matrix,
    )

    print(f"Metrics saved to: {file_path}")



