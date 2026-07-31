import argparse
import json
import os
import re
import numpy as np
import pandas as pd

from dataclasses import dataclass
from src.load_dataset import load_dataset
from src.model.metrics import ft_accuracy
from src.model.builder import network_builder
from src.preprocessing import get_target, min_max, z_score


@dataclass
class PredictConfig:
    dataset_path: str
    params_path: str
    model_path: str
    display_accuracy: bool
    output_file: str


def _get_predict_config(args: argparse.Namespace) -> PredictConfig:
    return PredictConfig(
        dataset_path=args.dataset,
        params_path=args.params_path,
        model_path=args.model_path,
        display_accuracy=args.display_accuracy,
        output_file=args.output_file
    )


def _display_prediction_accuracy(df: pd.DataFrame, y_pred: np.ndarray):
    y_true = get_target(df).T
    accuracy = ft_accuracy(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}%")


def _retrieve_params(
        params_path: str
) -> tuple[tuple[np.ndarray], tuple[np.ndarray]]:
    params = np.load(params_path)

    keys_W = [key for key in params.files if key.startswith("W")]
    keys_W.sort(key=lambda k: int(re.search(r'\d+', k).group()))
    W = [params[key] for key in keys_W]

    keys_b = [key for key in params.files if key.startswith("b")]
    keys_b.sort(key=lambda k: int(re.search(r'\d+', k).group()))
    b = [params[key] for key in keys_b]

    return tuple(W), tuple(b)


def _retrieve_model_config(
        model_path: str
) -> tuple[tuple[int], str, list[str], dict]:
    with open(model_path, "r") as file:
        model_json = json.load(file)

        return (
            model_json["layers"],
            model_json["activation"],
            model_json["features"],
            model_json["normalization"]
        )


def _save_pred_to_file(y_pred: np.ndarray, output_file: str):
    # Build diagnosis column
    y_pred_classes = y_pred.argmax(axis=0)
    diagnosis = np.where(y_pred_classes == 1, "M", "B")

    # Create dir if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write to file
    pd.DataFrame(
        {"diagnosis": diagnosis}
    ).to_csv(output_file, index=False)


def _preprocessing(
        df: pd.DataFrame,
        features: list[str],
        normalization_json: dict
) -> np.ndarray:
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
    config = _get_predict_config(args)

    # Retrieve model's architecture, features and normalization data
    layers, activation, features, normalization_json = _retrieve_model_config(
        config.model_path
    )

    # Retrieve network parameters
    W, b = _retrieve_params(config.params_path)

    # Build X ndarray from DataFrame to predict
    df = load_dataset(config.dataset_path)
    X = _preprocessing(df, features, normalization_json)

    # Create Neuron Network
    network = network_builder(layers, activation)

    # Initialize network parameters
    network.load_parameters(W, b)

    # Predict
    y_pred = network.predict(X)

    # Accuracy
    if config.display_accuracy and "diagnosis" in df.columns:
        _display_prediction_accuracy(df, y_pred)

    # Save Prediction to file
    _save_pred_to_file(y_pred, config.output_file)
