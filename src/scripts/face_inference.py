"""Face inference step definition."""

import argparse
import subprocess
import typing
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

import utils


class ModelBase(ABC):
    """Base class for model loaders."""

    @abstractmethod
    def _prepare(self) -> None:
        """Prepare the model for inference."""
        ...

    @abstractmethod
    def predict(self, image_data: np.ndarray) -> np.ndarray:
        """
        Run inference on the input image data.

        Parameters:
        -----------
        image_data : np.ndarray
            The input image data to run inference on.

        Returns:
        --------
        np.ndarray
            The output of the model after running inference.
        """
        ...


class FrameworkName(Enum):
    """Enum for recipe names."""

    TFLITE = "tflite"


class TFLiteModel(ModelBase):
    """Class for loading and running TFLite models."""

    def __init__(self, model_path: str) -> None:
        """
        Initialize the TFLite model loader.

        Parameters:
        -----------
        model_path : str
            The path to the TFLite model file.
        """
        from ai_edge_litert.interpreter import Interpreter

        self._model = Interpreter(model_path=model_path)
        self._prepare()

    def _prepare(self) -> None:
        """Prepare the model for inference."""
        self._model.allocate_tensors()
        self.input_details = self._model.get_input_details()
        self.output_details = self._model.get_output_details()

    @staticmethod
    def _preprocess(image_data: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image data.

        Parameters:
        -----------
        image_data : np.ndarray
            The input image data to preprocess.

        Returns:
        --------
        np.ndarray
            The preprocessed image data.
        """
        import tensorflow as tf

        image_data = tf.convert_to_tensor(image_data, dtype=tf.float32)
        image_data = tf.image.resize(image_data, [112, 112])
        image_data = image_data / 127.5 - 1.0
        image_data = tf.expand_dims(image_data, axis=0)

        return image_data.numpy()

    def predict(self, image_data: np.ndarray) -> np.ndarray:
        """
        Run inference on the input image data.

        Parameters:
        -----------
        image_data : np.ndarray
            The input image data to run inference on.

        Returns:
        --------
        np.ndarray
            The output of the model after running inference.
        """
        try:
            image_data = self._preprocess(image_data)
            self._model.set_tensor(self.input_details[0]["index"], image_data)
            self._model.invoke()
            output = self._model.get_tensor(self.output_details[0]["index"])
            return output

        except Exception as e:
            output_shape = self.output_details[0]["shape"]
            output_dtype = self.output_details[0]["dtype"]
            return np.zeros(output_shape, dtype=output_dtype)


model_loader_collection = {
    FrameworkName.TFLITE.value: TFLiteModel,
}


def inference_faces_entrypoint(
    base_directory: Path,
    model_name: str,
) -> None:
    """
    Entry point for the face inference step.

    Parameters:
    -----------
    base_directory : Path
        The base directory inside the container where the detections will be saved.
    model_name : str
        The name of the normalization algorithm to use.

    Returns:
    -------
    None
    """
    base_directory_path = Path(base_directory)
    output_base_path = base_directory_path / "output"
    output_base_path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["sudo", "chmod", "-R", "777", output_base_path])

    predictions = _inference_faces(
        model=_load_model(base_directory_path / "model" / model_name),
        face_data=utils.read_face_image_data(base_directory_path / "input"),
    )
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(output_base_path / f"predictions_{model_name}.csv")


def _load_model(model_path: Path) -> ModelBase:
    """
    Load the face recognition model.

    Parameters:
    -----------
    model_name : Path
        The path to the model.

    Returns:
    --------
    ModelBase
        The loaded model.
    """
    model_loader = model_loader_collection.get(model_path.suffix.lstrip("."))
    return model_loader(str(model_path))


def _inference_faces(
    model: ModelBase,
    face_data: typing.Generator[utils.FaceImageData, None, None],
) -> typing.Generator[pd.DataFrame, None, None]:
    """
    Run inference on the face data using the specified model.

    Parameters:
    -----------
    model : ModelBase
        The model to use for inference.
    face_data : typing.Generator[utils.FaceImageData, None, None]
        The face data to run inference on.

    Returns:
    --------
    typing.Generator[pd.DataFrame, None, None]
        A generator that yields DataFrames containing the predictions.
    """

    for image_path, image_data in face_data:
        predictions = model.predict(image_data=image_data)

        predictions_df = pd.DataFrame(predictions)
        predictions_df.insert(0, "image_path", image_path)

        yield predictions_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Normalize faces in a dataset.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the normalization algorithm to use.",
    )
    args, _ = parser.parse_known_args()

    inference_faces_entrypoint(
        base_directory="/opt/ml/processing",
        model_name=args.model_name,
    )
