import pandas as pd

from src.constants import COLUMNS


def load_dataset(
    dataset_path: str, shuffle: bool = False, seed: int | None = None
) -> pd.DataFrame:
    """
    Load a WDBC dataset CSV, applying column names if the file has no header.
    """
    df = pd.read_csv(dataset_path, header=None, nrows=1)

    if df.iloc[0, 1] in ("M", "B"):
        df = pd.read_csv(dataset_path, header=None)
        df.columns = COLUMNS
    else:
        df = pd.read_csv(dataset_path)

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df
