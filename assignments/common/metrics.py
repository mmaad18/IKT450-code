import numpy as np

from numpy.typing import NDArray
from matplotlib import pyplot as plt

"""
Metrics
"""


"""
* TP: True Positive
* TN: True Negative
* FP: False Positive
* FN: False Negative
* MSE: Mean Square Error
* RMSE: Root Mean Square Error
"""
def _confusion_matrix(Y_val: NDArray[np.float64], Y_pred: NDArray[np.float64]) -> tuple[int, int, int, int]:
    tp = 1
    tn = 1
    fp = 1
    fn = 1

    for i in range(len(Y_val)):
        if Y_val[i] == 1.0 and Y_pred[i] == 1.0:
            tp += 1
        elif Y_val[i] == 0.0 and Y_pred[i] == 0.0:
            tn += 1
        elif Y_val[i] == 0.0 and Y_pred[i] == 1.0:
            fp += 1
        elif Y_val[i] == 1.0 and Y_pred[i] == 0.0:
            fn += 1

    return tp, tn, fp, fn


def _precision(TP: int, FP: int) -> float:
    return TP / (TP + FP)


def _recall(TP: int, FN: int) -> float:
    return TP / (TP + FN)


def _f1_score(precision: float, recall: float) -> float:
    return 2 * (precision * recall) / (precision + recall)


def _accuracy(TP: int, TN: int, FP: int, FN: int) -> float:
    return (TP + TN) / (TP + TN + FP + FN)


def evaluate_metrics(
        Y_val: NDArray[np.float64],
        Y_pred: NDArray[np.float64]
) -> NDArray[np.float64]:
    TP, TN, FP, FN = _confusion_matrix(Y_val, Y_pred)
    precision_score = _precision(TP, FP)
    recall_score = _recall(TP, FN)
    f1_score = _f1_score(precision_score, recall_score)
    accuracy_score = _accuracy(TP, TN, FP, FN)

    error = Y_pred - Y_val
    MSE = float(np.mean(error ** 2))
    RMSE = np.sqrt(MSE)

    return np.array([TP, TN, FP, FN, precision_score, recall_score, f1_score, accuracy_score, MSE, RMSE])


def plot_evaluation(evaluation: NDArray[np.float64], x_label: str, title_append: str="") -> None:
    _, axs = plt.subplots(3, 1, figsize=(10, 15))  # pyright: ignore [reportUnknownMemberType]

    evaluation_size = evaluation.shape[0]

    # Plot TP, TN, FP, FN
    axs[0].plot(range(1, evaluation_size + 1), evaluation[:, 1], label="TN")
    axs[0].plot(range(1, evaluation_size + 1), evaluation[:, 0], label="TP")
    axs[0].plot(range(1, evaluation_size + 1), evaluation[:, 3], label="FN")
    axs[0].plot(range(1, evaluation_size + 1), evaluation[:, 2], label="FP")
    axs[0].set_title("Confusion Matrix Components" + title_append, fontsize=20)
    axs[0].set_xlabel(x_label, fontsize=16)
    axs[0].set_ylabel("Count", fontsize=16)
    axs[0].legend(fontsize=16)
    axs[0].tick_params(labelsize=16)
    axs[0].grid(True)

    # Plot Precision, Recall, F1 Score, Accuracy
    axs[1].plot(range(1, evaluation_size + 1), evaluation[:, 7], label="Accuracy")
    axs[1].plot(range(1, evaluation_size + 1), evaluation[:, 4], label="Precision")
    axs[1].plot(range(1, evaluation_size + 1), evaluation[:, 6], label="F1 Score")
    axs[1].plot(range(1, evaluation_size + 1), evaluation[:, 5], label="Recall")
    axs[1].set_title("Performance Metrics" + title_append, fontsize=20)
    axs[1].set_xlabel(x_label, fontsize=16)
    axs[1].set_ylabel("Score", fontsize=16)
    axs[1].legend(fontsize=16)
    axs[1].tick_params(labelsize=16)
    axs[1].grid(True)

    # Plot MSE and RMSE
    axs[2].plot(range(1, evaluation_size + 1), evaluation[:, 9], label="RMSE")
    axs[2].plot(range(1, evaluation_size + 1), evaluation[:, 8], label="MSE")
    axs[2].set_title("Error Metrics" + title_append, fontsize=20)
    axs[2].set_xlabel(x_label, fontsize=16)
    axs[2].set_ylabel("Error", fontsize=16)
    axs[2].legend(fontsize=16)
    axs[2].tick_params(labelsize=16)
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()  # pyright: ignore [reportUnknownMemberType]



