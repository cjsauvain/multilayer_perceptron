# PYTHON_ARGCOMPLETE_OK

import sys

from jsonschema import ValidationError
from src.parsing.mlp.parsing import parse_command_line
from src.predict import predict
from src.split import split_dataset
from src.train import train


def main():
    """Parse CLI arguments and dispatch to the selected mode's handler."""
    try:
        args = parse_command_line()

        if args.mode == "train":
            train(args)
        elif args.mode == "predict":
            predict(args)
        elif args.mode == "split-dataset":
            split_dataset(args)

    except ValidationError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    except (KeyError, ValueError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
