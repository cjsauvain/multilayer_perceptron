import argparse


def parse_predict_command_line(subparsers: argparse._SubParsersAction):
    predict_parser = subparsers.add_parser(
        "predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    predict_parser.add_argument(
        "dataset",
        help="Dataset to predict",
        type=str
    )
    predict_parser.add_argument(
        "--params-path",
        help="Path of the file that contains weights + biases after training",
        type=str,
        default="params.npz"
    )
    predict_parser.add_argument(
        "--model-path",
        help=(
            "Path of the file that contains the model's architecture, "
            "features and normalization data"
        ),
        type=str,
        default="model.json"
    )
    predict_parser.add_argument(
        "--display-accuracy",
        help="Display the accuracy of the prediction",
        action="store_true",
        default=False
    )
    predict_parser.add_argument(
        "--output-file",
        help="Path of the output file",
        type=str,
        default="predictions/dataset_predict.csv"
    )
