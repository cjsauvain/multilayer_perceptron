import argparse
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.load_dataset import load_dataset
from src.model.builder import network_builder
from src.model.loss import binary_cross_entropy
from src.model.metrics import ft_accuracy
from src.model.schema import validate_model_schema
from src.preprocessing import get_target, min_max, z_score


@dataclass
class PredictConfig:
    dataset_path: str
    params_path: str
    model_path: str
    display_accuracy: bool
    output_file: str
    layers: tuple[int, ...]
    activation: str
    features: list[str]
    normalization: dict


def _get_predict_config(args: argparse.Namespace) -> PredictConfig:
    """Validate the model file and build a PredictConfig from it and CLI arguments."""
    model = validate_model_schema(args.model_path)

    return PredictConfig(
        dataset_path=args.dataset,
        params_path=args.params_path,
        model_path=args.model_path,
        display_accuracy=args.display_accuracy,
        output_file=args.output_file,
        layers=model["layers"],
        activation=model["activation"],
        features=model["features"],
        normalization=model["normalization"],
    )


def _retrieve_params(
    params_path: str,
) -> tuple[tuple[np.ndarray], tuple[np.ndarray]]:
    """
    Load and order the per-layer weight and bias arrays from an .npz file.
    """
    params = np.load(params_path)

    keys_W = [key for key in params.files if key.startswith("W")]
    keys_W.sort(key=lambda k: int(re.search(r"\d+", k).group()))
    W = [params[key] for key in keys_W]

    keys_b = [key for key in params.files if key.startswith("b")]
    keys_b.sort(key=lambda k: int(re.search(r"\d+", k).group()))
    b = [params[key] for key in keys_b]

    return tuple(W), tuple(b)


def _save_pred_to_file(y_pred: np.ndarray, output_file: str):
    """
    Convert predicted class probabilities to B/M labels and write them to CSV.
    """
    # Build diagnosis column
    y_pred_classes = y_pred.argmax(axis=0)
    diagnosis = np.where(y_pred_classes == 1, "M", "B")

    # Create dir if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write to file
    pd.DataFrame({"diagnosis": diagnosis}).to_csv(output_file, index=False)
    print(f"\nPredictions saved to '{output_file}'")


def _preprocessing(
    df: pd.DataFrame, features: list[str], normalization_json: dict
) -> np.ndarray:
    """
    Apply the stored normalization (z-score or min-max) to the input features.
    """
    X = np.ndarray(())

    if normalization_json["mode"] == "z-score":
        means = normalization_json["means"]
        stds = normalization_json["stds"]
        X, _, _ = z_score(df[features], means, stds)
    elif normalization_json["mode"] == "min-max":
        mins = normalization_json["mins"]
        maxs = normalization_json["maxs"]
        X, _, _ = min_max(df[features], mins, maxs)

    return X.T


def predict(args: argparse.Namespace):
    """
    Run the full prediction pipeline: load model/params, predict, save results.
    """
    # Validate the model and build the prediction config
    config = _get_predict_config(args)

    # Retrieve network parameters
    W, b = _retrieve_params(config.params_path)

    # Build X ndarray from DataFrame to predict
    df = load_dataset(config.dataset_path)
    X = _preprocessing(df, config.features, config.normalization)

    # Create Neuron Network
    network = network_builder(config.layers, config.activation)

    # Initialize network parameters
    network.load_parameters(W, b)

    # Predict
    y_pred = network.predict(X)

    # Evaluate
    diagnosis_valid = (
        "diagnosis" in df.columns and not df["diagnosis"].isna().any()
    )
    if diagnosis_valid:
        y_true = get_target(df).T
        # Loss
        print(f"Loss: {binary_cross_entropy(y_true, y_pred)}")
        # Accuracy
        if config.display_accuracy:
            print(f"Accuracy: {ft_accuracy(y_true, y_pred):.2f}%")

    # Save Prediction to file
    _save_pred_to_file(y_pred, config.output_file)
