
import numpy as np

from assignments.utils import display_info


'''
Functions
'''
def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_prime(X):
    return sigmoid(X) * (1 - sigmoid(X))


def softmax(X):
    expX = np.exp(X)
    return expX / np.sum(expX)


def hidden_layer(W, X, phi, phi_d):
    V = X @ W
    return phi(V), phi_d(V)


def output_layer(W, X, phi):
    V = X @ W
    return phi(V)


def network_forward(Ws, X):
    Ys = []
    Yds = []
    for i in range(len(Ws) - 1):
        X, Yd = hidden_layer(Ws[i], X, sigmoid, sigmoid_prime)
        Ys.append(X)
        Yds.append(Yd)

    X = output_layer(Ws[-1], X, softmax)
    Ys.append(X)

    return Ys, Yds


def network_backward(Ws, Ys, Yds, T, eta):
    E = T - Ys[-1]
    d_L = E
    D_W_L = eta * Ys[-2].T @ d_L
    Ws[-1] += D_W_L

    for i in range(len(Ws) - 2, -1, -1):
        d_L = d_L @ Ws[i + 1].T  * Yds[i]
        D_W_L = eta * Ys[i].T @ d_L
        Ws[i] += D_W_L

    return Ws


'''
Unit tests
'''
def unit_tests():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    W = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    V = X @ W

    sigmoid_test = sigmoid(V)
    print(sigmoid_test)

    softmax_test = softmax(V)
    print(softmax_test)

    layer_test = hidden_layer(W, X, sigmoid, sigmoid_prime)
    print(layer_test)

    Ws = [W, W, W]

    Ys, Yds = network_forward(Ws, X)
    print(Ys)
    print(Yds)

    T = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)

    network_backward_test = network_backward(Ws, Ys, Yds, T, 0.1)
    print(network_backward_test)


def main():
    display_info(2)

    unit_tests()


main()
