import argparse


def _ratio_between_0_and_1(value: str) -> float:
    """Parse a CLI argument as a float strictly greater than 0 and up to 1."""
    f = float(value)

    if not 0 < f < 1:
        raise argparse.ArgumentTypeError(
            f"training ratio must be in the range ]0, 1[, got {f}"
        )

    return f


def parse_split_command_line(subparsers: argparse._SubParsersAction):
    """Register the `split-dataset` subcommand and its arguments."""
    split_parser = subparsers.add_parser(
        "split-dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    split_parser.add_argument("dataset", help="Dataset to split")
    split_parser.add_argument(
        "--train-ratio",
        help="Train ratio used to split the dataset",
        type=_ratio_between_0_and_1,
        default=0.8,
    )
    split_parser.add_argument(
        "--training-dataset",
        help="Name of the resulting training dataset",
        default="datasets/dataset_training.csv",
    )
    split_parser.add_argument(
        "--validation-dataset",
        help="Name of the resulting validation dataset",
        default="datasets/dataset_validation.csv",
    )
    split_parser.add_argument(
        "--seed",
        help="Seed to use to shuffle the dataset",
        type=int,
        default=42,
    )
