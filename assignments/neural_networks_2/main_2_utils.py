import numpy as np


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

    return X_train, X_val, Y_train.reshape(-1, 1), Y_val.reshape(-1, 1), filtered_data

