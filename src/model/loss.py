import numpy as np


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean binary cross-entropy loss over the batch."""
    m = y_true.shape[1]
    epsilon = 1e-15

    y = y_true[0]
    p = np.clip(y_pred[0], epsilon, 1 - epsilon)

    return (-1 / m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
