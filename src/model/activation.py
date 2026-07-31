import numpy as np


def sigmoid(prev_A: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    Z = W.dot(prev_A) + b
    A = 1 / (1 + np.exp(-Z))

    return A


def sigmoid_deriv(A: np.ndarray):
    return A * (1 - A)


def relu(prev_A: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    Z = W.dot(prev_A) + b
    A = np.maximum(0, Z)

    return A


def relu_deriv(A: np.ndarray) -> np.ndarray:
    return (A > 0).astype(float)


def softmax(prev_A: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    Z = W.dot(prev_A) + b
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    Z_exp = np.exp(Z_stable)
    sum_exp = np.sum((Z_exp), axis=0, keepdims=True)

    A = Z_exp / sum_exp

    return A
