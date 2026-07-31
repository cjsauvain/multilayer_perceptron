import numpy as np


def ft_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_classes = y_true.argmax(axis=0)
    y_pred_classes = y_pred.argmax(axis=0)

    accuracy = (
        (y_true_classes == y_pred_classes).sum()
        / y_true_classes.size
    ) * 100

    return accuracy
