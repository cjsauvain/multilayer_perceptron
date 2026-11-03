import argparse

from src.parsing.utils import positive_int


def parse_train_command_line(subparsers: argparse._SubParsersAction):
    """Register the `train` subcommand and its arguments."""
    train_parser = subparsers.add_parser(
        "train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    train_parser.add_argument("dataset_training", help="Training dataset")
    train_parser.add_argument("dataset_validation", help="Validation dataset")
    train_parser.add_argument(
        "--params-path",
        help="Path of the file that contains model's weights + biases",
        type=str,
        default="params.npz",
    )
    train_parser.add_argument(
        "--model-path",
        help=(
            "Path of the file that contains the model's architecture, "
            "features and normalization data"
        ),
        type=str,
        default="model.json",
    )
    train_parser.add_argument(
        "--learning-rate",
        help="Learning rate for gradient descent",
        type=float,
        default=0.1,
    )
    train_parser.add_argument(
        "--epochs",
        help="Number of training epochs",
        type=positive_int,
        default=10000,
    )
    train_parser.add_argument(
        "--disable-epochs-metrics",
        help="Disable displaying metrics for each epochs during training",
        action="store_true",
        default=False,
    )
    train_parser.add_argument(
        "--disable-visualization",
        help="Disable loss and accuracy visualization after training",
        action="store_true",
        default=False,
    )
    train_parser.add_argument(
        "--seed",
        help="Seed to use for weights initialization",
        type=int,
        default=None,
    )
    train_parser.add_argument(
        "--patience",
        help=(
            "Epochs without validation loss improvement before early stopping"
        ),
        type=positive_int,
        default=20,
    )
