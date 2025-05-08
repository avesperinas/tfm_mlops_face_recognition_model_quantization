"""Model quantization step definition."""

import argparse
import typing
from enum import Enum
from pathlib import Path

import ai_edge_quantizer
from ai_edge_quantizer import recipe


Recipe = list[dict[str, typing.Any]]


class RecipeName(Enum):
    """Enum for recipe names."""

    DYNAMIC_WI8_AFP32 = "dynamic_wi8_afp32"


quantization_recipes = {
    RecipeName.DYNAMIC_WI8_AFP32.value: recipe.dynamic_wi8_afp32(),
}


def quantize_model_entrypoint(
    base_directory: str,
    input_model_name: str,
    quantized_model_name: str,
    quantization_recipe_name: str,
) -> None:
    """
    Entry point for the quantization script.

    Parameters:
    ---------
    base_directory : str
        The base directory inside the container where the detections will be saved.
    input_model_name : str
        The name of the input model to be quantized.
    quantized_model_name : str
        The name of the quantized model to be saved.
    quantization_recipe_name : str
        Name of the quantization recipe to be used.

    Returns:
    -------
    None
    """
    base_directory_path = Path(base_directory)
    output_directory_path = base_directory_path / "output"
    output_directory_path.mkdir(parents=True, exist_ok=True)
    output_model_path = output_directory_path / quantized_model_name

    _quantize_model(
        quantizer=_get_quantizer(
            str(base_directory_path / "input" / input_model_name),
            quantization_recipe_name,
        ),
        quantized_model_path=output_model_path,
    )


def _get_quantizer(
    base_model_path: str,
    quantization_recipe_name: str,
) -> ai_edge_quantizer.Quantizer:
    """
    Get the quantizer object, parametrized with the defined recipe.

    Parameters:
    ---------
    base_model_path : str
        Path to the base model to be quantized.
    quantization_recipe_name : str
        Name of the quantization recipe to be used.

    Returns:
    -------
    quantizer: ai_edge_quantizer.Quantizer
        The quantizer object.
    """
    quantizer = ai_edge_quantizer.Quantizer(float_model=base_model_path)
    quantizer.load_quantization_recipe(
        recipe=quantization_recipes.get(quantization_recipe_name),
    )
    return quantizer


def _quantize_model(
    quantizer: ai_edge_quantizer.Quantizer,
    quantized_model_path: str,
) -> None:
    """
    Quantize the model using the provided quantizer.

    Parameters:
    ---------
    quantizer : ai_edge_quantizer.Quantizer
        The quantizer object.
    quantized_model_path : str
        Path to save the quantized model.

    Returns:
    -------
    None
    """
    quantization_result = quantizer.quantize()
    quantization_result.export_model(quantized_model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Normalize faces in a dataset.")
    parser.add_argument(
        "--input_model_name",
        type=str,
        required=True,
        help="Name of the input model to be quantized.",
    )
    parser.add_argument(
        "--quantized_model_name",
        type=str,
        required=True,
        help="Name of the quantized model to be saved.",
    )
    parser.add_argument(
        "--quantization_recipe_name",
        type=str,
        required=True,
        choices=list(quantization_recipes.keys()),
        help="Name of the quantization recipe to be used.",
    )
    args, _ = parser.parse_known_args()

    quantize_model_entrypoint(
        base_directory="/opt/ml/processing/",
        input_model_name=args.input_model_name,
        quantized_model_name=args.quantized_model_name,
        quantization_recipe_name=args.quantization_recipe_name,
    )
