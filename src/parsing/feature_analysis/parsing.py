import argparse
import argcomplete

from src.parsing.utils import positive_int


def parse_command_line() -> argparse.Namespace:
    """Build the feature_analysis.py CLI parser with its subcommands and arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset", help="Dataset to analyze")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Feature Selection
    selection_parser = subparsers.add_parser(
        "feature-selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Enable feature selection"
    )
    selection_parser.add_argument(
        "--features-number",
        help="Number of features to select",
        type=positive_int,
        default=3
    )

    # Feature Visualization
    selection_parser = subparsers.add_parser(
        "feature-visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Enable feature visualization individually"
    )
    selection_parser.add_argument(
        "--visualization-mode",
        help="Visualization method to use",
        choices=["scatter_plot", "boxplot"],
        default="scatter_plot"
    )

    argcomplete.autocomplete(parser)

    return parser.parse_args()
