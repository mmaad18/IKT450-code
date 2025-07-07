import unittest

import numpy as np

from assignments.neural_networks_2.main_2 import sigmoid, hidden_layer, network_forward, network_backward, sigmoid_prime

class TestMain2(unittest.TestCase):
    def test_sigmoid(self):
        X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)

        sigmoid_test = sigmoid(X)
        self.assertEqual(sigmoid_test.shape, X.shape)


    def test_hidden_layer(self):
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

        layer_test = hidden_layer(W, X, sigmoid, sigmoid_prime)
        self.assertEqual(layer_test[0].shape, X.shape)
        self.assertEqual(layer_test[1].shape, X.shape)


    def test_network_forward(self):
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

        Ws = [W, W, W]

        Ys, Yds = network_forward(Ws, X)
        self.assertEqual(len(Ys), len(Ws) + 1)
        self.assertEqual(len(Yds), len(Ws) - 1)


    def test_network_backward(self):
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

        Ws = [W, W, W]

        Ys, Yds = network_forward(Ws, X)

        T = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=float)

        network_backward_test = network_backward(Ws, Ys, Yds, T, 0.1, 0.01, 0.9)
        self.assertEqual(len(network_backward_test), len(Ws))


if __name__ == '__main__':
    unittest.main()

