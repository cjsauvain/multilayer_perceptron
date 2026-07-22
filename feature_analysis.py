import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from src.utils.utils import build_column_names


def _positive_int(value: str) -> int:
    i = int(value)

    if i <= 0:
        raise argparse.ArgumentTypeError(
            f"{value} must be a positive integer"
        )

    return i


def _parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        argument_default=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Feature Selection
    selection_parser = subparsers.add_parser(
        "feature-selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Enable feature selection"
    )
    selection_parser.add_argument(
        "--features-number",
        help= "Number of features to select",
        type= _positive_int,
        default= 3
    )
    
    # Feature Visualization
    selection_parser = subparsers.add_parser(
        "feature-visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Enable feature visualization individually"
    )
    selection_parser.add_argument(
        "--method",
        help= "Visualization method to use",
        choices=["scatter_plot", "boxplot"],
        default= "scatter_plot"
    )

    return parser.parse_args()


def _feature_selection(df: pd.DataFrame, features_number: int):
    df["diagnosis"] = (df["diagnosis"] == "M").astype('int')

    diagnosis_corr = df.corr()["diagnosis"].abs()

    relevant_features = diagnosis_corr[diagnosis_corr >= 0.5].index
    df = df[relevant_features]
    total_features = len(df.drop(columns="diagnosis").columns)

    corr = df.drop(columns=["diagnosis"]).corr().abs()
    feature_corr = corr[corr >= 0.7]
    mask = np.triu(np.ones(feature_corr.shape, dtype=bool))
    feature_corr = feature_corr.mask(mask).unstack().dropna()
    removed_features = []
    for (feat_a, feat_b) in feature_corr.index:
        nb_removed_features = len(removed_features)
        if total_features - nb_removed_features == features_number:
            break
        if feat_a not in removed_features and feat_b not in removed_features:
            feat_a_is_weaker = diagnosis_corr[feat_a] < diagnosis_corr[feat_b]
            removed_feature = feat_a if feat_a_is_weaker else feat_b
            removed_features.append(removed_feature)
    print(df.drop(columns=removed_features + ["diagnosis"]).columns)


def _feature_visualization(df: pd.DataFrame, method: str):
    df = pd.read_csv('datasets/data.csv')

    columns = build_column_names()
    df.columns = columns
    df = df.drop(columns=["ID", "diagnosis"])
    features = df.columns

    _, axes = plt.subplots(5, 6, figsize=(18, 9))

    if method == "boxplot":
        for ax, feature in zip(axes.ravel(), features):
            df.boxplot(column=feature, ax=ax)
            ax.set_title(feature)
    elif method == "scatter_plot":
        for ax, feature in zip(axes.ravel(), features):
            ax.scatter(df[feature], range(len(df)))
            ax.set_title(feature)

    plt.tight_layout()
    plt.show()

def main():
    args = _parse_command_line()

    df = pd.read_csv(args.dataset, header=None)
    columns = build_column_names()
    df.columns = columns

    if args.mode == "feature-selection":
        _feature_selection(df, args.features_number)
    if args.mode == "feature-visualization":
        _feature_visualization(df, args.method)


if __name__ == "__main__":
    main()