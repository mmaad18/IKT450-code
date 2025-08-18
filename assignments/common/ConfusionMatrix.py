from dataclasses import dataclass, field
from numpy.typing import NDArray

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class ConfusionMatrix:
    size: int
    labels: list[str]
    matrix: NDArray[np.int32] = field(init=False)
    metrics_size: int = 11

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


    def MCC(self) -> float:
        c = np.sum(self.matrix.diagonal())
        s = np.sum(self.matrix)

        p = np.sum(self.matrix, axis=0)
        t = np.sum(self.matrix, axis=1)

        sum1 = np.sum(p * t)
        sum2 = np.sum(p**2)
        sum3 = np.sum(t**2)

        return (c * s - sum1) / np.sqrt((s**2 - sum2) * (s**2 - sum3))


    def cohens_kappa(self) -> float:
        c = np.sum(self.matrix.diagonal())
        s = np.sum(self.matrix)

        p = np.sum(self.matrix, axis=0)
        t = np.sum(self.matrix, axis=1)

        sum1 = np.sum(p * t)

        return (c * s - sum1) / (s**2 - sum1)


    def plot(self, title_append: str="") -> None:
        plt.figure(figsize=(16, 16))
        sns.heatmap(
            self.matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=self.labels,
            yticklabels=self.labels,
            annot_kws={"size": 24}
        )
        plt.title("Confusion Matrix" + title_append, fontsize=30)
        plt.xlabel("Predicted Label", fontsize=24)
        plt.ylabel("True Label", fontsize=24)
        plt.xticks(fontsize=20, rotation=45, ha="right")
        plt.yticks(fontsize=20, rotation=0)
        plt.tight_layout()
        plt.show()


    def metrics(self) -> NDArray[np.float64]:
        macro_average_precision = self.macro_average_precision()
        macro_average_recall = self.macro_average_recall()
        macro_f1_score = self.macro_f1_score()
        micro_average_precision = self.micro_average_precision()
        micro_average_recall = self.micro_average_recall()
        micro_f1_score = self.micro_f1_score()
        accuracy = self.accuracy()
        balanced_accuracy = self.balanced_accuracy()
        balanced_accuracy_weighted = self.balanced_accuracy_weighted()
        MCC = self.MCC()
        cohens_kappa = self.cohens_kappa()

        return np.array([
            macro_average_precision,
            macro_average_recall,
            macro_f1_score,
            micro_average_precision,
            micro_average_recall,
            micro_f1_score,
            accuracy,
            balanced_accuracy,
            balanced_accuracy_weighted,
            MCC,
            cohens_kappa
        ])


    def plot_metrics(self, history: NDArray[np.float64], title_append: str = "") -> None:
        epochs = np.arange(history.shape[0])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

        # --- Macro ---
        axes[0, 0].plot(epochs, history[:, 0], label="Macro Precision")
        axes[0, 0].plot(epochs, history[:, 1], label="Macro Recall")
        axes[0, 0].plot(epochs, history[:, 2], label="Macro F1")
        axes[0, 0].set_title("Macro Metrics")
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # --- Micro ---
        axes[0, 1].plot(epochs, history[:, 3], label="Micro Precision")
        axes[0, 1].plot(epochs, history[:, 4], label="Micro Recall")
        axes[0, 1].plot(epochs, history[:, 5], label="Micro F1")
        axes[0, 1].set_title("Micro Metrics")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # --- Accuracy / Balanced ---
        axes[1, 0].plot(epochs, history[:, 6], label="Accuracy")
        axes[1, 0].plot(epochs, history[:, 7], label="Balanced Acc.")
        axes[1, 0].plot(epochs, history[:, 8], label="Balanced Acc. (w)")
        axes[1, 0].set_title("Accuracy Metrics")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # --- MCC / Kappa ---
        axes[1, 1].plot(epochs, history[:, 9], label="MCC")
        axes[1, 1].plot(epochs, history[:, 10], label="Cohen's κ")
        axes[1, 1].set_title("Agreement Metrics")
        axes[1, 1].set_ylim(-1, 1)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        for ax in axes[1]:
            ax.set_xlabel("Epoch")

        fig.suptitle(f"Metrics over Epochs{(' - ' + title_append) if title_append else ''}", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.show()


    def plotly_plot(self, title_append: str = "") -> None:
        fig = px.imshow(
            self.matrix,
            color_continuous_scale="Blues",
            text_auto=True,
            x=self.labels,
            y=self.labels,
            aspect="equal",
        )
        fig.update_layout(
            title=f"Confusion Matrix{title_append}",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=900,
            height=900,
        )
        fig.update_xaxes(side="top")
        fig.show()


    def plotly_plot_metrics(self, history: NDArray[np.float64], title_append: str = "") -> None:
        epochs = np.arange(history.shape[0])
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Macro Metrics", "Micro Metrics", "Accuracy Metrics", "Agreement Metrics"),
            shared_xaxes=True
        )

        fig.add_trace(go.Scatter(x=epochs, y=history[:, 0], name="Macro Precision"), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history[:, 1], name="Macro Recall"), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history[:, 2], name="Macro F1"), row=1, col=1)

        fig.add_trace(go.Scatter(x=epochs, y=history[:, 3], name="Micro Precision"), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=history[:, 4], name="Micro Recall"), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=history[:, 5], name="Micro F1"), row=1, col=2)

        fig.add_trace(go.Scatter(x=epochs, y=history[:, 6], name="Accuracy"), row=2, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history[:, 7], name="Balanced Acc."), row=2, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history[:, 8], name="Balanced Acc. (w)"), row=2, col=1)

        fig.add_trace(go.Scatter(x=epochs, y=history[:, 9], name="MCC"), row=2, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=history[:, 10], name="Cohen's κ"), row=2, col=2)

        fig.update_yaxes(range=[0, 1], row=1, col=1)
        fig.update_yaxes(range=[0, 1], row=1, col=2)
        fig.update_yaxes(range=[0, 1], row=2, col=1)
        fig.update_yaxes(range=[-1, 1], row=2, col=2)

        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)

        fig.update_layout(
            title=f"Metrics over Epochs{(' - ' + title_append) if title_append else ''}",
            width=1100,
            height=750,
            legend_tracegroupgap=6,
        )
        fig.show()



