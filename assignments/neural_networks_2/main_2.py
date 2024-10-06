import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from assignments.neural_networks_2.main_2_utils import data_preprocessing, plot_rms
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

    X_train, X_val, Y_train, Y_val, filtered_data = data_preprocessing("assignments/neural_networks_2/ecoli.data")

    Ws = [
        xavier_normal(8, 256),
        xavier_normal(256, 128),
        xavier_normal(128, 100),
        xavier_normal(100, 1)
    ]

    D_W_Ls = copy.deepcopy(Ws)

    eta = 0.001
    alpha = 0.9
    batch_size = 25

    RMSs = []

    for e in range(1000):
        for i in range(0, len(X_train), batch_size):
            Ws = train(X_train[i:i+batch_size], Y_train[i:i+batch_size], Ws, D_W_Ls, eta, alpha)

        Ys, _ = network_forward(Ws, X_val)

        error = Ys[-1] - Y_val
        RMS = np.sqrt(np.mean(error**2))
        RMSs.append(RMS)

    plot_rms(RMSs, eta, alpha, batch_size)


main()



