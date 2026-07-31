import numpy as np
import pandas as pd


def min_max(
        df: pd.DataFrame,
        mins: np.ndarray | None = None,
        maxs: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(df)

    X_mins = X.min(axis=0) if mins is None else mins
    X_maxs = X.max(axis=0) if maxs is None else maxs

    safe_diff = np.where(X_maxs - X_mins == 0, 1, X_maxs - X_mins)

    return (X - X_mins) / safe_diff, X_mins, X_maxs


def z_score(
        df: pd.DataFrame,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(df)

    X_means = X.mean(axis=0) if means is None else means
    X_stds = X.std(axis=0) if stds is None else stds

    safe_stds = np.where(X_stds == 0, 1.0, X_stds)

    return (X - X_means) / safe_stds, X_means, X_stds


def get_target(df: pd.DataFrame) -> np.ndarray:
    y_df = df["diagnosis"]

    # First column: 0="M", 1="B"
    # Second column: 0="B", 1="M"
    y = pd.get_dummies(y_df).astype(int).to_numpy()

    return y
