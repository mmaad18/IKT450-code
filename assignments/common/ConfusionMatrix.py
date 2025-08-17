from dataclasses import dataclass, field
from numpy.typing import NDArray

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


@dataclass
class ConfusionMatrix:
    size: int
    matrix: NDArray[np.int32] = field(init=False)

    def __post_init__(self):
        self.matrix = np.zeros((self.size, self.size), dtype=np.int32)


    def update(self, true_index: int, predicted_index: int):
        self.matrix[true_index, predicted_index] += 1


    def full_update(self, matrix: NDArray[np.int32]):
        self.matrix = matrix


    def actual_total(self) -> NDArray[np.int32]:
        return np.sum(self.matrix, axis=1)


    def precision(self, label_index: int) -> float:
        true_positive = self.matrix[label_index,label_index]
        false_positive = np.sum(self.matrix[:, label_index]) - true_positive
        return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0


    def accuracy(self) -> float:
        correct_predictions = np.trace(self.matrix)
        total_predictions = np.sum(self.matrix)
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0


    def balanced_accuracy(self) -> float:
        nominator = np.sum(self.matrix.diagonal() / self.actual_total())
        denominator = self.matrix.shape[0]
        return nominator / denominator if denominator > 0 else 0.0


    def balanced_accuracy_weighted(self) -> float:
        actual_total = self.actual_total()
        weights = actual_total / np.sum(actual_total)
        nominator = np.sum((self.matrix.diagonal() / actual_total) * weights)
        denominator = self.matrix.shape[0] * np.sum(weights)
        return nominator / denominator if denominator > 0 else 0.0


    def plot(self, title: str = "Confusion Matrix"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=range(self.size), yticklabels=range(self.size))
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()


