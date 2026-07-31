import argparse


def positive_int(value: str) -> int:
    i = int(value)

    if i <= 0:
        raise argparse.ArgumentTypeError(
            f"{value} must be a positive integer"
        )

    return i
