import unittest

import numpy as np

from assignments.common.ConfusionMatrix import ConfusionMatrix


class MyTestCase(unittest.TestCase):
    matrix = np.array([
        [6, 0, 1, 2],   # actual class 'a'
        [3, 9, 1, 1],   # actual class 'b'
        [1, 0, 10, 2],  # actual class 'c'
        [1, 2, 1, 12],  # actual class 'd'
    ])

    confusion_matrix = ConfusionMatrix(matrix.size)
    confusion_matrix.full_update(matrix)

    def test_actual_total(self):
        asserted = self.confusion_matrix.actual_total()
        expected = np.array([9, 14, 13, 16])

        np.testing.assert_array_equal(asserted, expected)


    def test_accuracy(self):
        asserted = self.confusion_matrix.accuracy()
        expected = (6 + 9 + 10 + 12) / 52

        self.assertEqual(asserted, expected)


    def test_balanced_accuracy(self):
        asserted = self.confusion_matrix.balanced_accuracy()
        expected = (6/9 + 9/14 + 10/13 + 12/16) / 4

        self.assertEqual(asserted, expected)


    def test_balanced_accuracy_weighted(self):
        asserted = self.confusion_matrix.balanced_accuracy_weighted()

        recalls = np.array([6/9, 9/14, 10/13, 12/16])
        weights = np.array([9/52, 14/52, 13/52, 16/52])
        W = np.sum(weights)

        expected = np.sum(recalls * weights) / (4 * W)

        self.assertEqual(asserted, expected)


if __name__ == '__main__':
    unittest.main()
