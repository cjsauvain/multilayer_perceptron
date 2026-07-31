import argparse

from src.parsing.utils import positive_int
from src.constants import FEATURES


def parse_train_command_line(subparsers: argparse._SubParsersAction):
    train_parser = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    train_parser.add_argument(
        "dataset_training",
        help="Training dataset"
    )
    train_parser.add_argument(
        "dataset_validation",
        help="Validation dataset"
    )
    train_parser.add_argument(
        "--hidden-layers",
        help="Hidden layers dimensions, e.g. --hidden-layers 24 24 24",
        type=positive_int,
        nargs="+",
        default=[16, 16]
    )
    train_parser.add_argument(
        "--features",
        help="Training features, e.g. --features radius_mean texture_se",
        nargs="+",
        choices=FEATURES,
        metavar="FEATURE",
        default=[
            "perimeter_worst",
            "compactness_worst",
            "concavity_worst",
            "concave_points_worst"
        ]
    )
    train_parser.add_argument(
        "--params-path",
        help="Path of the file that contains weights + biases after training",
        type=str,
        default="params.npz"
    )
    train_parser.add_argument(
        "--model-path",
        help=(
            "Path of the file that contains the model's architecture, "
            "features and normalization data"
        ),
        type=str,
        default="model.json"
    )
    train_parser.add_argument(
        "--activation",
        help="Hidden layers activation function, e.g. --activation sigmoid",
        choices=["sigmoid", "relu"],
        default="relu"
    )
    train_parser.add_argument(
        "--weights-initializer",
        help="Weight initializer (advise: He for ReLU, Glorot for sigmoid)",
        choices=["HeNormal", "GlorotNormal"],
        default="HeNormal"
    )
    train_parser.add_argument(
        "--learning-rate",
        help="Learning rate for gradient descent",
        type=float,
        default=0.1
    )
    train_parser.add_argument(
        "--epochs",
        help="Number of training epochs",
        type=positive_int,
        default=10000
    )
    train_parser.add_argument(
        "--normalization",
        help="Normalization method",
        choices=["min-max", "z-score"],
        default="z-score"
    )
    train_parser.add_argument(
        "--disable-epochs-metrics",
        help="Disable displaying metrics for each epochs during training",
        action="store_true",
        default=False
    )
    train_parser.add_argument(
        "--disable-visualization",
        help="Disable loss and accuracy visualization after training",
        action="store_true",
        default=False
    )
    train_parser.add_argument(
        "--seed",
        help="Seed to use for weights initialization",
        type=int,
        default=None
    )
    train_parser.add_argument(
        "--patience",
        help=(
            "Epochs without validation loss improvement "
            "before early stopping"
        ),
        type=positive_int,
        default=5
    )
