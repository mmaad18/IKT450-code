import copy

import numpy as np

from assignments.common.metrics import evaluate_metrics, plot_evaluation
from assignments.neural_networks_2.main_2_utils import data_preprocessing, shuffle_data
from numpy.typing import NDArray
from typing import Callable

"""
Functions
"""
def xavier_normal(input_size: int, layer_size: int) -> NDArray[np.float64]:
    std_dev = np.sqrt(2.0 / (input_size + layer_size))
    return np.random.normal(loc=0.0, scale=std_dev, size=(input_size, layer_size))


def sigmoid(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-X))


def sigmoid_prime(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return sigmoid(X) * (1 - sigmoid(X))


def hidden_layer(
        W: NDArray[np.float64],
        X: NDArray[np.float64],
        phi: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        phi_d: Callable[[NDArray[np.float64]], NDArray[np.float64]]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    V = X @ W
    Y = phi(V)
    Yd = phi_d(V)

    Y[:, -1] = 1
    Yd[:, -1] = 0

    return Y, Yd


def output_layer(
        W: NDArray[np.float64],
        X: NDArray[np.float64],
        phi: Callable[[NDArray[np.float64]], NDArray[np.float64]]
) -> NDArray[np.float64]:
    return phi(X @ W)


def network_forward(
        Ws: list[NDArray[np.float64]],
        X: NDArray[np.float64]
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    Ys: list[NDArray[np.float64]] = [X]
    Yds: list[NDArray[np.float64]] = []

    for i in range(len(Ws) - 1):
        X, Yd = hidden_layer(Ws[i], X, sigmoid, sigmoid_prime)  # pyright: ignore [reportConstantRedefinition]

        Ys.append(X)
        Yds.append(Yd)

    X = output_layer(Ws[-1], X, sigmoid)  # pyright: ignore [reportConstantRedefinition]
    Ys.append(X)

    return Ys, Yds


def network_backward(
        Ws: list[NDArray[np.float64]],
        D_W_Ls: list[NDArray[np.float64]],
        Ys: list[NDArray[np.float64]],
        Yds: list[NDArray[np.float64]],
        T: NDArray[np.float64],
        eta: float,
        alpha: float
) -> list[NDArray[np.float64]]:
    E = T - Ys[-1]
    d_L = E
    D_W_L = eta * Ys[-2].T @ d_L
    Ws[-1] += D_W_L + alpha * D_W_Ls[-1]
    D_W_Ls[-1] = D_W_L

    for i in range(len(Ws) - 2, -1, -1):
        d_L = (d_L @ Ws[i + 1].T) * Yds[i]
        D_W_L = eta * Ys[i].T @ d_L  # pyright: ignore [reportConstantRedefinition]
        Ws[i] += D_W_L + alpha * D_W_Ls[i]
        D_W_Ls[i] = D_W_L

    return Ws


def train(
        X: NDArray[np.float64],
        T: NDArray[np.float64],
        Ws: list[NDArray[np.float64]],
        D_W_Ls: list[NDArray[np.float64]],
        eta: float,
        alpha: float
) -> list[NDArray[np.float64]]:
    Ys, Yds = network_forward(Ws, X)
    Ws = network_backward(Ws, D_W_Ls, Ys, Yds, T, eta, alpha)

    return Ws


def main() -> None:
    X_train, X_val, Y_train, Y_val = data_preprocessing("assignments/neural_networks_2/ecoli.data")

    # Settings lists
    eta_list = [0.001, 0.001, 0.001]
    alpha_list = [0.9, 1.35, 1.8]
    batch_size_list = [10, 10, 10]

    evaluation_list: list[NDArray[np.float64]] = []

    for s in range(len(eta_list)):
        eta = eta_list[s]
        alpha = alpha_list[s]
        batch_size = batch_size_list[s]

        Ws = [
            xavier_normal(8, 3 + 1),
            xavier_normal(3 + 1, 2 + 1),
            xavier_normal(2 + 1, 1)
        ]

        D_W_Ls = copy.deepcopy(Ws)

        evaluation = np.zeros((1000, 10))

        for e in range(0, 1000):
            for i in range(0, len(X_train), batch_size):
                Ws = train(X_train[i:i+batch_size], Y_train[i:i+batch_size], Ws, D_W_Ls, eta, alpha)

            Ys, _ = network_forward(Ws, X_val)

            Y_pred_bin = (Ys[-1] > 0.5).astype(np.float64)
            evaluation[e] = evaluate_metrics(Y_val, Y_pred_bin)

            X_train, X_val, Y_train, Y_val = shuffle_data(X_train, X_val, Y_train, Y_val, e)

        evaluation_list.append(evaluation)

    idx = 1
    plot_evaluation(evaluation_list[idx], "Epoch", f" (η={eta_list[idx]}, α={alpha_list[idx]}, batch_size={batch_size_list[idx]})")


main()



