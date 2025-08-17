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


    def test_precision(self):
        asserted = self.confusion_matrix.precision()
        expected = np.array([6 / 11, 9 / 11, 10 / 13, 12 / 17])

        np.testing.assert_array_equal(asserted, expected)


    def test_recall(self):
        asserted = self.confusion_matrix.recall()
        expected = np.array([6 / 9, 9 / 14, 10 / 13, 12 / 16])

        np.testing.assert_array_equal(asserted, expected)


    def test_macro_average_precision(self):
        asserted = self.confusion_matrix.macro_average_precision()

        precision = np.array([6 / 11, 9 / 11, 10 / 13, 12 / 17])
        expected = np.mean(precision)

        self.assertEqual(asserted, expected)


    def test_macro_average_recall(self):
        asserted = self.confusion_matrix.macro_average_recall()
        recall = np.array([6 / 9, 9 / 14, 10 / 13, 12 / 16])
        expected = np.mean(recall)

        self.assertEqual(asserted, expected)


    def test_macro_f1_score(self):
        asserted = self.confusion_matrix.macro_f1_score()
        macro_average_precision = np.mean(np.array([6 / 11, 9 / 11, 10 / 13, 12 / 17]))
        macro_average_recall = np.mean(np.array([6 / 9, 9 / 14, 10 / 13, 12 / 16]))
        expected = 2 * (macro_average_precision * macro_average_recall) / (macro_average_precision + macro_average_recall)

        self.assertEqual(asserted, expected)


    def test_micro_average_precision(self):
        asserted = self.confusion_matrix.micro_average_precision()
        precision = np.array([6 / 11, 9 / 11, 10 / 13, 12 / 17])
        expected = np.sum(precision) / 52

        self.assertEqual(asserted, expected)


    def test_micro_average_recall(self):
        asserted = self.confusion_matrix.micro_average_recall()
        recall = np.array([6 / 9, 9 / 14, 10 / 13, 12 / 16])
        expected = np.sum(recall) / 52

        self.assertEqual(asserted, expected)


    def test_micro_f1_score(self):
        asserted = self.confusion_matrix.micro_f1_score()
        micro_average_precision = np.sum(np.array([6 / 11, 9 / 11, 10 / 13, 12 / 17])) / 52
        micro_average_recall = np.sum(np.array([6 / 9, 9 / 14, 10 / 13, 12 / 16])) / 52
        expected = 2 * (micro_average_precision * micro_average_recall) / (micro_average_precision + micro_average_recall)

        self.assertEqual(asserted, expected)


    def test_MCC(self):
        asserted = self.confusion_matrix.MCC()

        c = 6 + 9 + 10 + 12
        s = 52

        sum1 = 9 * 11 + 14 * 11 + 13 * 13 + 16 * 17
        sum2 = 11**2 + 11**2 + 13**2 + 17**2
        sum3 = 9**2 + 14**2 + 13**2 + 16**2

        expected = (c * s - sum1) / np.sqrt((s**2 - sum2) * (s**2 - sum3))

        self.assertEqual(asserted, expected)


    def test_cohens_kappa(self):
        asserted = self.confusion_matrix.cohens_kappa()

        c = 6 + 9 + 10 + 12
        s = 52

        sum1 = 9 * 11 + 14 * 11 + 13 * 13 + 16 * 17

        expected = (c * s - sum1) / (s**2 - sum1)


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
