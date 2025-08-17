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


    def update_vector(self, Y_val: NDArray[np.int32], Y_pred: NDArray[np.int32], reset: bool=False):
        if len(Y_val) != len(Y_pred):
            raise ValueError("Length of true labels and predicted labels must be the same.")

        if reset:
            self.matrix = np.zeros((self.size, self.size), dtype=np.int32)

        K = self.matrix.shape[0]
        idx = Y_val * K + Y_pred
        cm = np.bincount(idx, minlength=K * K).reshape(K, K)
        self.matrix += cm


    def full_update(self, matrix: NDArray[np.int32]):
        self.matrix = matrix


    def actual_total(self) -> NDArray[np.int32]:
        return np.sum(self.matrix, axis=1)


    def grand_total(self) -> int:
        return np.sum(self.matrix)


    def precision(self) -> NDArray[np.float64]:
        return np.diag(self.matrix) / np.sum(self.matrix, axis=0)


    def recall(self) -> NDArray[np.float64]:
        return np.diag(self.matrix) / np.sum(self.matrix, axis=1)


    def macro_average_precision(self) -> float:
        return np.mean(self.precision())


    def macro_average_recall(self) -> float:
        return np.mean(self.recall())


    def macro_f1_score(self) -> float:
        precision = self.macro_average_precision()
        recall = self.macro_average_recall()
        return 2 * (precision * recall) / (precision + recall)


    def micro_average_precision(self) -> float:
        return np.sum(self.precision()) / self.grand_total()


    def micro_average_recall(self) -> float:
        return np.sum(self.recall()) / self.grand_total()


    def micro_f1_score(self) -> float:
        precision = self.micro_average_precision()
        recall = self.micro_average_recall()
        return 2 * (precision * recall) / (precision + recall)


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


    def MCC(self):
        c = np.sum(self.matrix.diagonal())
        s = np.sum(self.matrix)

        p = np.sum(self.matrix, axis=0)
        t = np.sum(self.matrix, axis=1)

        sum1 = np.sum(p * t)
        sum2 = np.sum(p**2)
        sum3 = np.sum(t**2)

        return (c * s - sum1) / np.sqrt((s**2 - sum2) * (s**2 - sum3))


    def cohens_kappa(self):
        c = np.sum(self.matrix.diagonal())
        s = np.sum(self.matrix)

        p = np.sum(self.matrix, axis=0)
        t = np.sum(self.matrix, axis=1)

        sum1 = np.sum(p * t)

        return (c * s - sum1) / (s**2 - sum1)


    def plot(self, title: str = "Confusion Matrix"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=range(self.size), yticklabels=range(self.size))
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()


