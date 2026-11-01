import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.load_dataset import load_dataset
from src.model.builder import network_builder
from src.model.schema import validate_model_schema
from src.preprocessing import get_target, min_max, z_score
from src.visualization import visualization


@dataclass
class TrainConfig:
    dataset_training: str
    dataset_validation: str
    layers: tuple[int, ...]
    features: list[str]
    activation: str
    weights_initializer: str
    learning_rate: float
    epochs: int
    disable_epochs_metrics: bool
    disable_visualization: bool
    normalization: str
    params_path: str
    model_path: str
    seed: int | None
    patience: int


def _get_train_config(args: argparse.Namespace) -> TrainConfig:
    """
    Build a TrainConfig by merging CLI arguments
    with the validated model.json.
    """
    model = validate_model_schema(args.model_path)

    return TrainConfig(
        dataset_training=args.dataset_training,
        dataset_validation=args.dataset_validation,
        layers=model["layers"],
        features=model["features"],
        activation=model["activation"],
        weights_initializer=model["weights_initializer"],
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        disable_epochs_metrics=args.disable_epochs_metrics,
        disable_visualization=args.disable_visualization,
        normalization=model["normalization"]["mode"],
        params_path=args.params_path,
        model_path=args.model_path,
        seed=args.seed,
        patience=args.patience,
    )


def _save_model_to_file(
    model_path: str,
    layers: tuple[int, ...],
    activation: str,
    features: list[str],
    weights_initializer: str,
    normalization: dict,
):
    """
    Write the model architecture, features and
    normalization data to model_path.
    """
    json_model = {
        "layers": layers,
        "activation": activation,
        "features": features,
        "weights_initializer": weights_initializer,
        "normalization": normalization["normalization"],
    }

    with open(model_path, "w") as file:
        json.dump(json_model, file, indent=4)


def _preprocessing(
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    normalization_method: str,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Normalize train/validation features and one-hot encode their targets."""
    X_train = np.array(())
    X_valid = np.array(())
    y_train = get_target(training_df)
    y_valid = get_target(validation_df)

    json_normalization = {"normalization": {}}

    # Z-Score
    if normalization_method == "z-score":
        X_train, means, stds = z_score(training_df[features])
        X_valid, _, _ = z_score(validation_df[features], means, stds)
        json_normalization["normalization"]["mode"] = "z-score"
        json_normalization["normalization"]["means"] = means.tolist()
        json_normalization["normalization"]["stds"] = stds.tolist()

    # Min-Max
    elif normalization_method == "min-max":
        X_train, mins, maxs = min_max(training_df[features])
        X_valid, _, _ = min_max(validation_df[features], mins, maxs)
        json_normalization["normalization"]["mode"] = "min-max"
        json_normalization["normalization"]["mins"] = mins.tolist()
        json_normalization["normalization"]["maxs"] = maxs.tolist()

    print("\nX_train shape:", X_train.T.shape)
    print("y_train shape:", y_train.T.shape)
    print("X_valid shape:", X_valid.T.shape)
    print("y_valid shape:", y_valid.T.shape, "\n")

    return X_train.T, y_train.T, X_valid.T, y_valid.T, json_normalization


def train(args: argparse.Namespace):
    """
    Run the full training pipeline: preprocess, fit the network, save results.
    """
    config = _get_train_config(args)

    # Build Training + Validation DataFrames
    training_df = load_dataset(config.dataset_training)
    validation_df = load_dataset(config.dataset_validation)

    # Pre-processing
    # No imputation needed: No missing attribute values
    X_train, y_train, X_valid, y_valid, json_normalization = _preprocessing(
        training_df,
        validation_df,
        config.normalization,
        config.features,
    )

    # Save model's architecture, features and normalization data to file
    _save_model_to_file(
        config.model_path,
        config.layers,
        config.activation,
        config.features,
        config.weights_initializer,
        json_normalization,
    )

    # Create Neuron Network
    if config.seed is not None:
        np.random.seed(config.seed)

    network = network_builder(
        config.layers, config.activation, config.weights_initializer
    )

    # Fit
    network.fit(
        X_train,
        y_train,
        X_valid,
        y_valid,
        config.epochs,
        config.disable_epochs_metrics,
        config.learning_rate,
        config.patience,
    )
    # Training Stopping Reason
    print(f"\nTraining Stopping Reason: {network.stopping_reason}")
    # Best Epoch
    print(f"\nBest  epoch: {network.best_epoch}")
    # Best Loss
    print(f"Valid loss: {network.loss_valid[-1]:.4f}")
    # Display Train + Valid Accuracy
    print(f"Train accuracy: {network.acc_train[-1]:.2f}%")
    print(f"Valid accuracy: {network.acc_valid[-1]:.2f}%\n")

    # Save parameters W + b to .npz file
    network.save_parameters(config.params_path)

    # Visualization
    if config.disable_visualization is False:
        visualization(network)
