import argparse
import argcomplete

from src.parsing.mlp.train_parsing import parse_train_command_line
from src.parsing.mlp.predict_parsing import parse_predict_command_line
from src.parsing.mlp.split_parsing import parse_split_command_line


def parse_command_line():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    parse_train_command_line(subparsers)
    parse_predict_command_line(subparsers)
    parse_split_command_line(subparsers)

    argcomplete.autocomplete(parser)

    return parser.parse_args()
