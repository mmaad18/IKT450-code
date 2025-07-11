import numpy as np
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


