
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

    return X_train, X_val, Y_train, Y_val, dataset


def indicator_function(y, j):
    return 1 if y == j else 0


def classify(X_train, Y_train, x0, K):
    distances = np.linalg.norm(X_train - x0, axis=1)

    nearest_neighbors_indices = np.argsort(distances)[:K]

    probabilities = {}

    for label in set(Y_train):
        probability = 0

        for index in nearest_neighbors_indices:
            probability += indicator_function(label, Y_train[index])

        probabilities[label] = probability / K

    return max(probabilities, key=probabilities.get)


'''
* TP: True Positive
* TN: True Negative
* FP: False Positive
* FN: False Negative
* MSE: Mean Square Error
* RMSE: Root Mean Square Error
'''


def precision(TP, FP):
    return TP / (TP + FP)


def recall(TP, FN):
    return TP / (TP + FN)


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)


def mean_square_error(Y_true, Y_pred):
    return np.mean((Y_true - Y_pred) ** 2)


def root_mean_square_error(Y_true, Y_pred):
    return np.sqrt(mean_square_error(Y_true, Y_pred))


def evaluation(Y_val, Y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(Y_val)):
        if Y_val[i] == 1 and Y_pred[i] == 1:
            TP += 1
        elif Y_val[i] == 0 and Y_pred[i] == 0:
            TN += 1
        elif Y_val[i] == 0 and Y_pred[i] == 1:
            FP += 1
        elif Y_val[i] == 1 and Y_pred[i] == 0:
            FN += 1

    precision_score = precision(TP, FP)
    recall_score = recall(TP, FN)
    f1_score_value = f1_score(precision_score, recall_score)
    accuracy_score = accuracy(TP, TN, FP, FN)
    mse = mean_square_error(Y_val, Y_pred)
    rmse = root_mean_square_error(Y_val, Y_pred)

    print("Precision: ", precision_score)
    print("Recall: ", recall_score)
    print("F1 Score: ", f1_score_value)
    print("Accuracy: ", accuracy_score)
    print("Mean Square Error: ", mse)
    print("Root Mean Square Error: ", rmse)


def main():
    display_info(1)

    X_train, X_val, Y_train, Y_val, dataset = load_data("assignments/nearest-neighbour-1/pima-indians-diabetes.csv")

    K = 50

    Y_pred = []

    for i in range(len(X_val)):
        x0 = X_val[i]
        y_pred = classify(X_train, Y_train, x0, K)
        Y_pred.append(y_pred)

    evaluation(Y_val, Y_pred)


main()

