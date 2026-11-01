import json
from pathlib import Path

from jsonschema import validate


def validate_model_schema(model_path: str) -> dict:
    """
    Validate a model.json file against model.schema.json and return it.
    """
    with open(model_path, "r") as file:
        model = json.load(file)

    SCHEMA_PATH = Path(__file__).resolve().parents[2] / "model.schema.json"
    with open(SCHEMA_PATH, "r") as file:
        schema = json.load(file)

    validate(instance=model, schema=schema)

    if "layers" not in model:
        model["layers"] = [24, 24, 2]

    return model
