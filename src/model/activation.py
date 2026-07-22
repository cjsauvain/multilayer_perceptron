import numpy as np


def sigmoid(
        prev_A: np.ndarray,
        W: np.ndarray,
        b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    Z = W.dot(prev_A) + b
    A = 1 / np.exp(1 + Z)

    return A, Z


def sigmoid_deriv(A: np.ndarray):
    return A * (1 - A)


def relu(
        prev_A: np.ndarray,
        W: np.ndarray,
        b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    Z = W.dot(prev_A) + b
    A = np.maximum(0, Z)

    return A, Z


def relu_deriv(A: np.ndarray) -> np.ndarray:
    return (A > 0).astype(float)


def softmax():
    pass