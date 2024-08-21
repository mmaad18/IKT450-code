
import numpy as np

from assignments.utils import display_info


def load_data(file_path: str, split_ratio: float = 0.8):
    np.random.seed(7)

    # load pima indians dataset
    dataset = np.loadtxt(file_path, delimiter=",")
    np.random.shuffle(dataset)

    # split into input (X) and output (Y) variables
    index = int(len(dataset) * split_ratio)
    X_train = dataset[:index, 0:8]
    X_val = dataset[index:, 0:8]
    Y_train = dataset[:index, 8]
    Y_val = dataset[index:, 8]

    return X_train, X_val, Y_train, Y_val


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def indicator_function(x, y):
    return 1 if x == y else 0


def get_neighbours(X_train, Y_train, x, k):
    distances = []

    for i in range(len(X_train)):
        distances.append((i, distance(X_train[i], x), Y_train[i]))


'''
* TP: True Positive
* TN: True Negative
* FP: False Positive
* FN: False Negative
'''
def precision(TP, FP):
    return TP / (TP + FP)


def recall(TP, FN):
    return TP / (TP + FN)


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)


def main():
    display_info(1)

    X_train, X_val, Y_train, Y_val = load_data("assignments/nearest-neighbour-1/pima-indians-diabetes.csv")


main()
