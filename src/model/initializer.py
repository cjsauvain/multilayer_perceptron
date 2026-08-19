import numpy as np


def he_normal_initializer(n_in: int, n_out: int) -> np.ndarray:
    """Draw a weight matrix scaled for ReLU layers (He normal initialization)."""
    return np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)


def glorot_normal_initializer(n_in: int, n_out: int) -> np.ndarray:
    """Draw a weight matrix scaled for sigmoid/softmax layers (Glorot normal initialization)."""
    n_avg = (n_in + n_out) / 2

    return np.random.randn(n_out, n_in) * np.sqrt(2 / n_avg)
