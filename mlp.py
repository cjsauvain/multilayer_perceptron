# PYTHON_ARGCOMPLETE_OK

import sys

from src.parsing.mlp.parsing import parse_command_line
from src.split import split_dataset
from src.train import train
from src.predict import predict


def main():
    try:
        args = parse_command_line()

        if args.mode == "train":
            train(args)
        elif args.mode == "predict":
            predict(args)
        elif args.mode == "split-dataset":
            split_dataset(args)

    except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    except (KeyError, ValueError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
