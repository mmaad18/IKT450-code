import numpy as np
from matplotlib import pyplot as plt


def data_preprocessing(file_path: str, seed: int = 15, split_ratio: float = 0.8):
    np.random.seed(seed)

    with open(file_path, 'r') as file:
        data_list = [line.strip().split() for line in file]

    # Filter out 'cp' and 'im' labels
    data = np.array(data_list)
    mask = (data[:, -1] == "cp") | (data[:, -1] == "im")
    filtered_data = data[mask]

    # Replace 'cp' with '1' and 'im' with '0' in the last column
    labels = filtered_data[:, -1]
    filtered_data[:, -1] = np.where(labels == "cp", 1.0, 0.0)

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

    X_train = np.hstack((X_train, ones_column_train))
    X_val = np.hstack((X_val, ones_column_val))

    return X_train, X_val, Y_train.reshape(-1, 1), Y_val.reshape(-1, 1)


def shuffle_data(X_train, X_val, Y_train, Y_val, seed):
    np.random.seed(seed)

    # Shuffle training data
    train_indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[train_indices]
    Y_train_shuffled = Y_train[train_indices]

    # Shuffle validation data
    val_indices = np.random.permutation(X_val.shape[0])
    X_val_shuffled = X_val[val_indices]
    Y_val_shuffled = Y_val[val_indices]

    return X_train_shuffled, X_val_shuffled, Y_train_shuffled, Y_val_shuffled


def plot_loss(loss_type, losses, eta, alpha, batch_size):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label=f"{loss_type} over epochs", color='blue', linewidth=2)

    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel(f"{loss_type}", fontsize=16)
    plt.title(f"{loss_type} vs Epochs (eta={eta}, alpha={alpha}, batch_size={batch_size})", fontsize=20)

    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()


def plot_loss_multiple(loss_type, losses_list, eta_list, alpha_list, batch_size_list, size="1"):
    plt.figure(figsize=(10, 6))

    # Plot each RMS curve
    for i, RMSs in enumerate(losses_list):
        eta = eta_list[i]
        alpha = alpha_list[i]
        batch_size = batch_size_list[i]

        plt.plot(RMSs, label=f"{loss_type} (η={eta}, α={alpha}, batch_size={batch_size})", linewidth=2)

    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel(f"{loss_type}", fontsize=16)
    plt.title(f"{loss_type} vs Epochs, Size=({size})", fontsize=20)

    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()



