import argparse
import pandas as pd

from src.utils.utils import build_column_names


def split_dataset(args: argparse.Namespace):
    df = pd.read_csv(args.dataset, header=None)
    columns = build_column_names()
    df.columns = columns
    df = (
        df
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
        .drop(columns="ID")
    )


    split_index = int(args.train_ratio * len(df))
    training_df = df[:split_index]
    validation_df = df[split_index:]

    training_df.to_csv(args.training_dataset, index=False)
    validation_df.to_csv(args.validation_dataset, index=False)

