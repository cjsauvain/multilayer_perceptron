import argparse
from dataclasses import dataclass

import pandas as pd

from src.load_dataset import load_dataset


@dataclass
class SplitConfig:
    dataset_path: str
    training_dataset: str
    validation_dataset: str
    train_ratio: float
    seed: int | None


def _get_split_config(args: argparse.Namespace) -> SplitConfig:
    """Build a SplitConfig from parsed CLI arguments."""
    return SplitConfig(
        dataset_path=args.dataset,
        training_dataset=args.training_dataset,
        validation_dataset=args.validation_dataset,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


def split_dataset(args: argparse.Namespace):
    """
    Split a dataset into shuffled train/validation CSVs,
    stratified by diagnosis.
    """
    config = _get_split_config(args)

    df = load_dataset(config.dataset_path, shuffle=True, seed=config.seed)
    df_B = df[df["diagnosis"] == "B"].drop(columns="ID")
    df_M = df[df["diagnosis"] == "M"].drop(columns="ID")

    split_index_B = int(config.train_ratio * len(df_B))
    split_index_M = int(config.train_ratio * len(df_M))

    training_df = (
        pd.concat([df_B[:split_index_B], df_M[:split_index_M]])
        .sample(frac=1, random_state=config.seed)
        .reset_index(drop=True)
    )
    validation_df = (
        pd.concat([df_B[split_index_B:], df_M[split_index_M:]])
        .sample(frac=1, random_state=config.seed)
        .reset_index(drop=True)
    )

    training_df.to_csv(config.training_dataset, index=False)
    print(f"Training dataset saved to '{config.training_dataset}'")
    validation_df.to_csv(config.validation_dataset, index=False)
    print(f"Validation dataset saved to '{config.validation_dataset}'")
