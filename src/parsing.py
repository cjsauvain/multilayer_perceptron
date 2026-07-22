import argparse

from traitlets import default


def _ratio_between_0_and_1(value: str) -> float:
    f = float(value)

    if not 0 < f <= 1:
        raise argparse.ArgumentTypeError(
            f"training ratio must be in the range ]0, 1[, got {f}"
        )

    return f


def _positive_int(value: str) -> int:
    i = int(value)

    if i <= 0:
        raise argparse.ArgumentTypeError(
            f"{value} must be a positive integer"
        )

    return i


def _parse_train_command_line(subparsers: argparse._SubParsersAction):
    AVAILABLE_FEATURES = [
            "radius", "texture", "perimeter", "area", "smoothness", 
            "compactness", "concavity","concave_points","symmetry",
            "fractal_dimension"
    ]

    train_parser = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    train_parser.add_argument("dataset_training", help="Training dataset")
    train_parser.add_argument("dataset_validation", help="Validation dataset")
    train_parser.add_argument(
        "--layers",
        help="Hidden layers dimensions, positive integers, e.g. --layers 24 24 24",
        type=_positive_int,
        nargs="+",
        default=[16, 16, 16]
    )
    train_parser.add_argument(
        "--features",
        help="Training features, e.g. --features radius texture area",
        nargs="+",
        choices=AVAILABLE_FEATURES,
        default=AVAILABLE_FEATURES
    )
    train_parser.add_argument(
        "--activation",
        help="Hidden layers activation function, e.g. --activation sigmoid",
        choices=["sigmoid", "relu"],
        default="relu"
    )
    train_parser.add_argument(
        "--weights-initializer",
        help="Weight initializer: advised to use He with ReLU and Glorot with sigmoid",
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
        "--batch-ratio",
        help="Fraction of the training set per batch",
        type=_ratio_between_0_and_1,
        default=1
    )
    train_parser.add_argument(
        "--epochs",
        help="Number of training epochs",
        type=_positive_int,
        default=100
    )
    train_parser.add_argument(
        "--normalization",
        help="Normalization method",
        choices=["min-max", "z-score"],
        default="z-score"
    )


def _parse_predict_command_line(subparsers: argparse._SubParsersAction):
    predict_parser = subparsers.add_parser(
        "predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


def _parse_split_command_line(subparsers: argparse._SubParsersAction):
    split_parser = subparsers.add_parser(
        "split-dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    split_parser.add_argument("dataset", help="Dataset to split")
    split_parser.add_argument(
        "--train-ratio",
        help="Train ratio used to split the dataset",
        type=_ratio_between_0_and_1,
        default=0.8
    )
    split_parser.add_argument(
        "--training-dataset",
        help="Name of the resulting training dataset",
        default="datasets/dataset_training.csv"
    )
    split_parser.add_argument(
        "--validation-dataset",
        help="Name of the resulting validation dataset",
        default="datasets/dataset_validation.csv"
    )


def parsing_command_line():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    _parse_train_command_line(subparsers)
    _parse_predict_command_line(subparsers)
    _parse_split_command_line(subparsers)
    
    return parser.parse_args()