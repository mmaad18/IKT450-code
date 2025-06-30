import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def load_data(
        file_path: str,
        seed: int=7,
        split_ratio: float=0.8
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    np.random.seed(seed)

    # Load pima indians dataset
    dataset = np.loadtxt(file_path, delimiter=",")
    np.random.shuffle(dataset)

    # Split into input (X) and output (Y) variables
    idx = int(len(dataset) * split_ratio)
    X_train = dataset[:idx, 0:8]
    X_val = dataset[idx:, 0:8]
    Y_train = dataset[:idx, 8]
    Y_val = dataset[idx:, 8]

    return X_train, X_val, Y_train, Y_val, dataset


def plot_evaluation(evaluation: NDArray[np.float64], x_label: str, title_append: str="") -> None:
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    evaluation_size = evaluation.shape[0]

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

