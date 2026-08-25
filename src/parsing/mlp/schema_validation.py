import json

from jsonschema import validate


def validate_model_schema(
        model_path: str,
        return_model: bool = False
) -> dict | None:
    """
    Validate a model.json file against model.schema.json,
    optionally returning it.
    """
    with open(model_path, "r") as file:
        model = json.load(file)

    with open("model.schema.json", "r") as file:
        schema = json.load(file)

    validate(instance=model, schema=schema)

    if return_model:
        return model
    return None
