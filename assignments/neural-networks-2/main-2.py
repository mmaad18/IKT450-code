
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
    expX = np.exp(X - np.max(X, axis=1, keepdims=True))
    return expX / np.sum(expX, axis=1, keepdims=True)


def hidden_layer(W, X, phi, phi_d):
    ones_column = np.ones((X.shape[0], 1))

    X = np.hstack((ones_column, X))
    V = X @ W
    return phi(V), phi_d(V)


def output_layer(W, X, phi):
    V = X @ W
    return phi(V)


def network_forward(Ws, X):
    Ys = [1, X]
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
    Ws[-1] = D_W_L

    for i in range(len(Ws) - 2, -1, -1):
        d_L = (d_L @ Ws[i + 1].T) * Yds[i]
        D_W_L = eta * Ys[i].T @ d_L
        Ws[i] += D_W_L

    return Ws


def data_preprocessing(file_path: str, seed: int = 7, split_ratio: float = 0.8):
    np.random.seed(seed)

    with open(file_path, 'r') as file:
        data_list = [line.strip().split() for line in file]

    # Filter out 'cp' and 'im' labels
    data = np.array(data_list)
    mask = (data[:, -1] == 'cp') | (data[:, -1] == 'im')
    filtered_data = data[mask]

    # Replace 'cp' with '1' and 'im' with '0' in the last column
    labels = filtered_data[:, -1]
    filtered_data[:, -1] = np.where(labels == 'cp', 1.0, 0.0)

    np.random.shuffle(filtered_data)

    # Split into input (X) and output (Y) variables
    index = int(len(filtered_data) * split_ratio)
    X_train = filtered_data[:index, 1:8].astype(float)
    X_val = filtered_data[index:, 1:8].astype(float)
    Y_train = filtered_data[:index, 8].astype(float)
    Y_val = filtered_data[index:, 8].astype(float)

    # Add a column of ones to X_train and X_val
    n_train_samples = X_train.shape[0]
    n_val_samples = X_val.shape[0]

    ones_column_train = np.ones((n_train_samples, 1))
    ones_column_val = np.ones((n_val_samples, 1))

    X_train = np.hstack((ones_column_train, X_train))
    X_val = np.hstack((ones_column_val, X_val))

    return X_train, X_val, Y_train.reshape(-1, 1), Y_val.reshape(-1, 1), filtered_data


def train(X, T, Ws, eta):
    Ys, Yds = network_forward(Ws, X)
    Ws = network_backward(Ws, Ys, Yds, T, eta)

    return Ws


def main():
    display_info(2)

    #unit_tests()

    X_train, X_val, Y_train, Y_val, filtered_data = data_preprocessing("assignments/neural-networks-2/ecoli.data")

    Ws = [
        np.random.rand(8, 10),
        np.random.rand(10, 5),
        np.random.rand(5, 1)
    ]

    eta = 0.1
    batch_size = 10

    for e in range(10):
        for i in range(0, len(X_train), batch_size):
            Ws = train(X_train[i:i+batch_size], Y_train[i:i+batch_size], Ws, eta)

    Ys, _ = network_forward(Ws, X_val)

    a = 1


main()


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
