from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from numpy.typing import NDArray
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


    def _get_plotly_fig(self, title_append: str = "") -> go.Figure:
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
        return fig


    def _get_plotly_metrics_fig(self, history: NDArray[np.float64], title_append: str = "") -> go.Figure:
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
        return fig


    def plotly_plot(self, title_append: str = ""):
        fig = self._get_plotly_fig(title_append)
        fig.show()


    def plotly_plot_metrics(self, history: NDArray[np.float64], title_append: str = ""):
        fig = self._get_plotly_metrics_fig(history, title_append)
        fig.show()


    def save_plotly(self, save_path: Path, title_append: str = ""):
        fig = self._get_plotly_fig(title_append)
        fig.write_html(save_path)


    def save_plotly_metrics(self, history: NDArray[np.float64], save_path: Path, title_append: str = ""):
        fig = self._get_plotly_metrics_fig(history, title_append)
        fig.write_html(save_path)




