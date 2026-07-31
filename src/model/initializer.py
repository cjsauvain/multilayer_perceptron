import numpy as np


def he_normal_initializer(n_in: int, n_out: int) -> np.ndarray:
    return np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)


def glorot_normal_initializer(n_in: int, n_out: int) -> np.ndarray:
    n_avg = (n_in + n_out) / 2

    return np.random.randn(n_out, n_in) * np.sqrt(2 / n_avg)
