import argparse
import pandas as pd

from dataclasses import dataclass
from src.model.layer import DenseLayer
from src.model.network import Network
from src.preprocessing.preprocessing import normalize_data, get_target


@dataclass
class TrainConfig:
    dataset_training: str
    dataset_validation: str
    layers: tuple[int]
    features: tuple[str]
    activation: str
    weights_initializer: str
    learning_rate: float
    batch_ratio: float
    epochs: int
    normalization: str


def _parse_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        dataset_training=args.dataset_training,
        dataset_validation=args.dataset_validation,
        layers=args.layers,
        features=args.features,
        activation=args.activation,
        weights_initializer=args.weights_initializer,
        learning_rate=args.learning_rate,
        batch_ratio=args.batch_ratio,
        epochs=args.epochs,
        normalization=args.normalization
    )


def train(args: argparse.Namespace):
    config = _parse_args(args)

    training_df = pd.read_csv(config.dataset_training)
    validation_df = pd.read_csv(config.dataset_validation)

    ## Pre-processing
    # No imputation needed: No missing attribute values
    X_train = normalize_data(training_df, config.normalization)
    y_train = get_target(training_df)
    X_valid = normalize_data(validation_df, config.normalization)
    y_valid = get_target(validation_df)

    # Create Neuron Network
    denseLayers = tuple(
        DenseLayer(
            n_neurons=n_neurons,
            activation=config.activation,
            weight_initializer=config.weights_initializer
        ) for n_neurons in config.layers
    )

    network = Network(denseLayers)
    network.fit(X_train, y_train, X_valid, y_valid, config.epochs, config.learning_rate)
