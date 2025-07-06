
import numpy as np

from assignments.nearest_neighbour_1.main_1_utils import load_data, plot_evaluation
from utils import display_info
from numpy.typing import NDArray


def indicator_function(y: int, j: int) -> int:
    return 1 if y == j else 0


def manhattan_distances(X_train: NDArray[np.float64], x0: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sum(np.abs(X_train - x0), axis=1)


def euclidean_distances(X_train: NDArray[np.float64], x0: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.linalg.norm(X_train - x0, axis=1)


def chebyshev_distances(X_train: NDArray[np.float64], x0: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.max(np.abs(X_train - x0), axis=1)


'''
p is a parameter that determines the type of distance:
* p=1: Manhattan Distance
* p=2: Euclidean Distance
* p=inf: Chebyshev Distance
* Higher values of p give more emphasis to larger differences
'''
def minkowski_distances(X_train: NDArray[np.float64], x0: NDArray[np.float64], p: int=2) -> NDArray[np.float64]:
    return np.sum(np.abs(X_train - x0)**p, axis=1)**(1/p)


def classify(X_train: NDArray[np.float64], Y_train: NDArray[np.float64], x0: NDArray[np.float64], K: int) -> int:
    distances = chebyshev_distances(X_train, x0)

    nearest_neighbors_indices = np.argsort(distances)[:K]

    probabilities = {}

    for label in set(Y_train):
        probability = 0

        for idx in nearest_neighbors_indices:
            probability += indicator_function(label, int(Y_train[idx]))

        probabilities[label] = probability / K

    return max(probabilities, key=probabilities.get)  # pyright: ignore [reportCallIssue, reportUnknownMemberType, reportArgumentType, reportUnknownVariableType]


'''
* TP: True Positive
* TN: True Negative
* FP: False Positive
* FN: False Negative
* MSE: Mean Square Error
* RMSE: Root Mean Square Error
'''
def confusion_matrix(Y_val: NDArray[np.float64], Y_pred: NDArray[np.float64]) -> tuple[int, int, int, int]:
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(Y_val)):
        if Y_val[i] == 1 and Y_pred[i] == 1:
            tp += 1
        elif Y_val[i] == 0 and Y_pred[i] == 0:
            tn += 1
        elif Y_val[i] == 0 and Y_pred[i] == 1:
            fp += 1
        elif Y_val[i] == 1 and Y_pred[i] == 0:
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


def mean_square_error(Y_true: NDArray[np.float64], Y_pred: NDArray[np.float64]) -> float:
    return float(np.mean((Y_true - Y_pred) ** 2))


def root_mean_square_error(Y_true: NDArray[np.float64], Y_pred: NDArray[np.float64]) -> float:
    return np.sqrt(mean_square_error(Y_true, Y_pred))


def evaluate_K(
        X_train: NDArray[np.float64],
        X_val: NDArray[np.float64],
        Y_train: NDArray[np.float64],
        Y_val: NDArray[np.float64],
        K: int
) -> NDArray[np.float64]:
    Y_pred: NDArray[np.float64] = np.zeros(len(X_val), dtype=np.float64)

    for i in range(len(X_val)):
        x0 = X_val[i]
        y_pred = classify(X_train, Y_train, x0, K)
        Y_pred[i] = y_pred

    TP, TN, FP, FN = confusion_matrix(Y_val, Y_pred)

    precision_score = _precision(TP, FP)
    recall_score = _recall(TP, FN)
    f1_score = _f1_score(precision_score, recall_score)
    accuracy_score = _accuracy(TP, TN, FP, FN)
    MSE = mean_square_error(Y_val, Y_pred)
    RMSE = root_mean_square_error(Y_val, Y_pred)

    return np.array([TP, TN, FP, FN, precision_score, recall_score, f1_score, accuracy_score, MSE, RMSE])


def evaluate_K_given_seed(seed: int):
    X_train, X_val, Y_train, Y_val, _ = load_data("assignments/nearest_neighbour_1/pima-indians-diabetes.csv", seed=seed)
    K_evaluation_size = 150

    evaluation = np.zeros((K_evaluation_size, 10))

    for K in range(1, K_evaluation_size + 1):
        evaluation[K - 1] = evaluate_K(X_train, X_val, Y_train, Y_val, K)

    plot_evaluation(evaluation, "K", f" (Seed={seed}, distance=Chebyshev)")


def evaluate_seed_given_K(K: int):
    S_evaluation_size = 1000
    start = 900
    evaluation = np.zeros((S_evaluation_size-start, 10))

    for s in range(start, S_evaluation_size):
        np.random.seed(s)
        X_train, X_val, Y_train, Y_val, _ = load_data("assignments/nearest_neighbour_1/pima-indians-diabetes.csv", seed=s)
        evaluation[s-start] = evaluate_K(X_train, X_val, Y_train, Y_val, K)

    plot_evaluation(evaluation, f"Seed (Start: {start}, End: {S_evaluation_size})", f" (K={K})")


def main():
    display_info(1)

    evaluate_K_given_seed(7)


main()

