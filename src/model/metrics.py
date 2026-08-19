import numpy as np


def ft_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the percentage of correctly predicted classes."""
    y_true_classes = y_true.argmax(axis=0)
    y_pred_classes = y_pred.argmax(axis=0)

    accuracy = (
        (y_true_classes == y_pred_classes).sum()
        / y_true_classes.size
    ) * 100

    return accuracy


def ft_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute precision (%) for the malignant class: TP / (TP + FP)."""
    y_pred_classes = y_pred.argmax(axis=0)

    tp = np.sum((y_true[1] == 1) & (y_pred_classes == 1))
    fp = np.sum((y_true[1] == 0) & (y_pred_classes == 1))

    safe_denom = tp + fp if tp + fp != 0 else 1.0

    return tp / safe_denom * 100


def ft_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall (%) for the malignant class: TP / (TP + FN)."""
    y_pred_classes = y_pred.argmax(axis=0)

    tp = np.sum((y_true[1] == 1) & (y_pred_classes == 1))
    fn = np.sum((y_true[1] == 1) & (y_pred_classes == 0))

    safe_denom = tp + fn if tp + fn != 0 else 1.0

    return tp / safe_denom * 100


def ft_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the harmonic mean of precision and recall for the malignant class."""
    precision = ft_precision(y_true, y_pred)
    recall = ft_recall(y_true, y_pred)

    safe_denom = precision + recall if precision + recall != 0 else 1.0

    return 2 * precision * recall / safe_denom
