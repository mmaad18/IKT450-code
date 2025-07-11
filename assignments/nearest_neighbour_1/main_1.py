
import numpy as np

from assignments.common.metrics import evaluate_metrics, plot_evaluation
from assignments.nearest_neighbour_1.main_1_utils import load_data
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

    return evaluate_metrics(Y_val, Y_pred)


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

