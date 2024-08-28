
import numpy as np
import matplotlib.pyplot as plt

from assignments.utils import display_info


def load_data(file_path: str, seed: int = 7, split_ratio: float = 0.8):
    np.random.seed(seed)

    # Load pima indians dataset
    dataset = np.loadtxt(file_path, delimiter=",")
    np.random.shuffle(dataset)

    # Split into input (X) and output (Y) variables
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
def confusion_matrix(Y_val, Y_pred):
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

    return TP, TN, FP, FN


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


def evaluate_K(X_train, X_val, Y_train, Y_val, K):
    Y_pred = []

    for i in range(len(X_val)):
        x0 = X_val[i]
        y_pred = classify(X_train, Y_train, x0, K)
        Y_pred.append(y_pred)

    TP, TN, FP, FN = confusion_matrix(Y_val, Y_pred)

    precision_score = precision(TP, FP)
    recall_score = recall(TP, FN)
    f1_score_value = f1_score(precision_score, recall_score)
    accuracy_score = accuracy(TP, TN, FP, FN)
    MSE = mean_square_error(Y_val, Y_pred)
    RMSE = root_mean_square_error(Y_val, Y_pred)

    return np.array([TP, TN, FP, FN, precision_score, recall_score, f1_score_value, accuracy_score, MSE, RMSE])


def plot_evaluation(evaluation, evaluation_size, x_label, title_append=""):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot TP, TN, FP, FN
    axs[0].plot(range(1, evaluation_size + 1), evaluation[:, 1], label="TN")
    axs[0].plot(range(1, evaluation_size + 1), evaluation[:, 0], label="TP")
    axs[0].plot(range(1, evaluation_size + 1), evaluation[:, 3], label="FN")
    axs[0].plot(range(1, evaluation_size + 1), evaluation[:, 2], label="FP")
    axs[0].set_title("Confusion Matrix Components" + title_append, fontsize=20)
    axs[0].set_xlabel(x_label, fontsize=16)
    axs[0].set_ylabel("Count", fontsize=16)
    axs[0].legend(fontsize=16)
    axs[0].tick_params(labelsize=16)
    axs[0].grid(True)

    # Plot Precision, Recall, F1 Score, Accuracy
    axs[1].plot(range(1, evaluation_size + 1), evaluation[:, 7], label="Accuracy")
    axs[1].plot(range(1, evaluation_size + 1), evaluation[:, 4], label="Precision")
    axs[1].plot(range(1, evaluation_size + 1), evaluation[:, 6], label="F1 Score")
    axs[1].plot(range(1, evaluation_size + 1), evaluation[:, 5], label="Recall")
    axs[1].set_title("Performance Metrics" + title_append, fontsize=20)
    axs[1].set_xlabel(x_label, fontsize=16)
    axs[1].set_ylabel("Score", fontsize=16)
    axs[1].legend(fontsize=16)
    axs[1].tick_params(labelsize=16)
    axs[1].grid(True)

    # Plot MSE and RMSE
    axs[2].plot(range(1, evaluation_size + 1), evaluation[:, 9], label="RMSE")
    axs[2].plot(range(1, evaluation_size + 1), evaluation[:, 8], label="MSE")
    axs[2].set_title("Error Metrics" + title_append, fontsize=20)
    axs[2].set_xlabel(x_label, fontsize=16)
    axs[2].set_ylabel("Error", fontsize=16)
    axs[2].legend(fontsize=16)
    axs[2].tick_params(labelsize=16)
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_K_given_seed(seed):
    X_train, X_val, Y_train, Y_val, dataset = load_data("assignments/nearest-neighbour-1/pima-indians-diabetes.csv", seed=seed)
    K_evaluation_size = 150

    evaluation = np.zeros((K_evaluation_size, 10))

    for K in range(1, K_evaluation_size + 1):
        evaluation[K - 1] = evaluate_K(X_train, X_val, Y_train, Y_val, K)

    plot_evaluation(evaluation, K_evaluation_size, "K", f" (Seed={seed})")


def evaluate_seed_given_K(K):
    S_evaluation_size = 1000
    start = 900
    evaluation = np.zeros((S_evaluation_size-start, 10))

    for s in range(start, S_evaluation_size):
        np.random.seed(s)
        X_train, X_val, Y_train, Y_val, dataset = load_data("assignments/nearest-neighbour-1/pima-indians-diabetes.csv", seed=s)
        evaluation[s-start] = evaluate_K(X_train, X_val, Y_train, Y_val, K)

    plot_evaluation(evaluation, S_evaluation_size-start, f"Seed (Start: {start}, End: {S_evaluation_size})", f" (K={K})")


def main():
    display_info(1)

    evaluate_K_given_seed(7)


main()

