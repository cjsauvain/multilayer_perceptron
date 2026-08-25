# PYTHON_ARGCOMPLETE_OK

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from src.load_dataset import load_dataset
from src.parsing.feature_analysis.parsing import parse_command_line


def _feature_selection(df: pd.DataFrame, features_nb_to_keep: int):
    """
    Print the N features most correlated with
    diagnosis, pruning redundant pairs.
    """
    df["diagnosis"] = (df["diagnosis"] == "M").astype('int')

    diagnosis_corr = df.corr()["diagnosis"].abs()

    # Keep features with correlation >= 0.5 with diagnosis
    relevant_features = diagnosis_corr[diagnosis_corr >= 0.5].index
    df = df[relevant_features]
    nb_relevant_features = len(df.drop(columns="diagnosis").columns)

    # From relevant features, remove correlated features
    # and keep <features_nb_to_keep> features
    corr = df.drop(columns=["diagnosis"]).corr().abs()
    feature_corrs = corr[corr >= 0.7]
    # Mask upper triangle (+ diagonal) to keep only the lower triangle,
    # avoiding duplicate pairs (corr matrix is symmetric) and self-correlations
    mask = np.triu(np.ones(feature_corrs.shape, dtype=bool))
    feature_corr = feature_corrs.mask(mask).unstack().dropna()

    # Remove correlated features
    removed_features = []
    for (feat_a, feat_b) in feature_corr.index:
        nb_removed_features = len(removed_features)
        if nb_relevant_features - nb_removed_features == features_nb_to_keep:
            break
        if feat_a not in removed_features and feat_b not in removed_features:
            feat_a_is_weaker = diagnosis_corr[feat_a] < diagnosis_corr[feat_b]
            removed_feature = feat_a if feat_a_is_weaker else feat_b
            removed_features.append(removed_feature)

    # Display selected features
    print(df.drop(columns=removed_features + ["diagnosis"]).columns)


def _feature_visualization(df: pd.DataFrame, visualization_mode: str):
    """Plot every feature individually as a boxplot or scatter plot grid."""
    df = df.drop(columns=["ID", "diagnosis"])

    _, axes = plt.subplots(5, 6, figsize=(18, 9))
    features = df.columns

    # Visualization Mode
    if visualization_mode == "boxplot":
        for ax, feature in zip(axes.ravel(), features):
            df.boxplot(column=feature, ax=ax)
            ax.set_title(feature)
    elif visualization_mode == "scatter_plot":
        for ax, feature in zip(axes.ravel(), features):
            ax.scatter(df[feature], range(len(df)))
            ax.set_title(feature)

    plt.tight_layout()
    plt.show()


def main():
    """Parse CLI arguments and run the selected feature analysis mode."""
    try:
        args = parse_command_line()

        # Build DataFrame
        df = load_dataset(args.dataset)

        # Analysis Modes
        if args.mode == "feature-selection":
            _feature_selection(df, args.features_number)
        if args.mode == "feature-visualization":
            _feature_visualization(df, args.visualization_mode)

    except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    except (KeyError, ValueError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
