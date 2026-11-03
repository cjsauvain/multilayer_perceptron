# PYTHON_ARGCOMPLETE_OK

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.load_dataset import load_dataset
from src.parsing.feature_analysis.parsing import parse_command_line

MIN_DIAGNOSIS_CORRELATION = 0.3
MAX_FEATURE_CORRELATION = 0.9
RED = "\033[1;4;31m"
RESET = "\033[0m"


def _feature_selection(df: pd.DataFrame):
    """
    Print the most correlated features with diagnosis, pruning redundant pairs.
    """
    # Keep features with correlation >= 0.3 with diagnosis
    df = df.drop(columns="ID", errors="ignore")
    df["diagnosis"] = (df["diagnosis"] == "M").astype("int")
    diagnosis_corr = (
        df.corr()["diagnosis"]
        .abs()
        .drop(labels="diagnosis")
        .sort_values(ascending=False)
    )
    relevant_features = diagnosis_corr[
        diagnosis_corr >= MIN_DIAGNOSIS_CORRELATION
    ].index
    relevant_features_df = df[relevant_features]

    # From relevant features, remove most correlated features
    corr = relevant_features_df.corr().abs()

    # Filter out most correlated features
    kept = []
    for feature in relevant_features:
        redundant = any(
            corr.loc[feature, kept_feature] > MAX_FEATURE_CORRELATION
            for kept_feature in kept
        )
        if not redundant:
            kept.append(feature)

    # Display correlations and selected features
    print(
        f"\n{RED}Absolute correlation of each feature with diagnosis{RESET}\n"
    )
    print(diagnosis_corr.round(3))
    print(
        f"\n{RED}Most correlated features (> {MAX_FEATURE_CORRELATION}) excluding diagnosis{RESET}\n"
    )
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).stack().round(3).sort_values(ascending=False)
    print(pairs[pairs > MAX_FEATURE_CORRELATION])
    print(
        f"\n{RED}Features kept by absolute correlation with diagnosis{RESET}\n"
    )
    print(diagnosis_corr[kept].round(3))


def _feature_visualization(df: pd.DataFrame, visualization_mode: str):
    """Plot every feature individually as a boxplot or scatter plot grid."""
    df = df.drop(columns=["ID"], errors="ignore")
    features = df.drop(columns="diagnosis").columns

    _, axes = plt.subplots(5, 6, figsize=(18, 9))

    # Visualization Mode
    for ax, feature in zip(axes.ravel(), features):
        benign = df.loc[df["diagnosis"] == "B", feature]
        malignant = df.loc[df["diagnosis"] == "M", feature]

        if visualization_mode == "boxplot":
            ax.boxplot([benign, malignant])
            ax.set_xticks([1, 2], ["B", "M"])
        elif visualization_mode == "scatter_plot":
            ax.scatter(benign, range(len(benign)), s=8, c="tab:blue", label="B")
            ax.scatter(
                malignant, range(len(malignant)), s=8, c="tab:orange", label="M"
            )
            ax.legend(fontsize=7)

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
            _feature_selection(df)
        if args.mode == "feature-visualization":
            _feature_visualization(df, args.visualization_mode)

    except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
