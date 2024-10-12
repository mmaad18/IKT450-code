import copy

import numpy as np
from scipy.stats import norm

from assignments.neural_networks_2.main_2_utils import data_preprocessing, plot_loss, plot_loss_multiple, shuffle_data
from assignments.utils import display_info

"""
Functions
"""
def xavier_normal(input_size: int, layer_size: int):
    std_dev = np.sqrt(2.0 / (input_size + layer_size))
    normal_dist = norm(loc=0.0, scale=std_dev)
    return normal_dist.rvs(size=(input_size, layer_size))

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_prime(X):
    return sigmoid(X) * (1 - sigmoid(X))


def hidden_layer(W, X, phi, phi_d):
    V = X @ W
    Y = phi(V)
    Yd = phi_d(V)

    Y[:, -1] = 1
    Yd[:, -1] = 0

    return Y, Yd


def output_layer(W, X, phi):
    return phi(X @ W)


def network_forward(Ws, X):
    Ys = [X]
    Yds = []
    for i in range(len(Ws) - 1):
        X, Yd = hidden_layer(Ws[i], X, sigmoid, sigmoid_prime)

        Ys.append(X)
        Yds.append(Yd)

    X = output_layer(Ws[-1], X, sigmoid)
    Ys.append(X)

    return Ys, Yds


def network_backward(Ws, D_W_Ls, Ys, Yds, T, eta, alpha):
    E = T - Ys[-1]
    d_L = E
    D_W_L = eta * Ys[-2].T @ d_L
    Ws[-1] += D_W_L + alpha * D_W_Ls[-1]
    D_W_Ls[-1] = D_W_L

    for i in range(len(Ws) - 2, -1, -1):
        d_L = (d_L @ Ws[i + 1].T) * Yds[i]
        D_W_L = eta * Ys[i].T @ d_L
        Ws[i] += D_W_L + alpha * D_W_Ls[i]
        D_W_Ls[i] = D_W_L

    return Ws


def train(X, T, Ws, D_W_Ls, eta, alpha):
    Ys, Yds = network_forward(Ws, X)
    Ws = network_backward(Ws, D_W_Ls, Ys, Yds, T, eta, alpha)

    return Ws


def main():
    display_info(2)

    X_train, X_val, Y_train, Y_val = data_preprocessing("assignments/neural_networks_2/ecoli.data")

    # Settings lists
    eta_list = [0.001, 0.001, 0.001]
    alpha_list = [0.9, 1.35, 1.8]
    batch_size_list = [10, 10, 10]

    MSEs_list = []

    for s in range(len(eta_list)):
        eta = eta_list[s]
        alpha = alpha_list[s]
        batch_size = batch_size_list[s]

        Ws = [
            xavier_normal(8, 50 + 1),
            xavier_normal(50 + 1, 40 + 1),
            xavier_normal(40 + 1, 30 + 1),
            xavier_normal(30 + 1, 20 + 1),
            xavier_normal(20 + 1, 1)
        ]

        D_W_Ls = copy.deepcopy(Ws)

        MSEs = []

        for e in range(1000):
            for i in range(0, len(X_train), batch_size):
                Ws = train(X_train[i:i+batch_size], Y_train[i:i+batch_size], Ws, D_W_Ls, eta, alpha)

            Ys, _ = network_forward(Ws, X_val)

            error = Ys[-1] - Y_val
            MSE = np.mean(error**2)
            MSEs.append(MSE)

            X_train, X_val, Y_train, Y_val = shuffle_data(X_train, X_val, Y_train, Y_val, e)

        MSEs_list.append(MSEs)

    plot_loss_multiple("MSE", MSEs_list, eta_list, alpha_list, batch_size_list, size="50, 40, 30, 20, 1")


main()



