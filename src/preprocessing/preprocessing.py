import numpy as np
import pandas as pd


def _min_max(df: pd.DataFrame) -> np.ndarray:
    X = np.array(df)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    return (X - mins) / (maxs - mins)


def _z_score(df: pd.DataFrame) -> np.ndarray:
    X = np.array(df)
    means = X.mean(axis=0)
    stds = X.std(axis=0)

    return (X - means) / stds


def normalize_data(df: pd.DataFrame, normalization: str) -> np.ndarray:
    if normalization == "min-max":
        return _min_max(df.drop(columns="diagnosis"))
    return _z_score(df.drop(columns="diagnosis"))


def get_target(df: pd.DataFrame) -> np.ndarray:
    y = np.array((df["diagnosis"] == "M").astype(int))
    y = y.reshape(y.shape[0], -1)

    return y