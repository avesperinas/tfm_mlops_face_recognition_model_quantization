"""Face detection step definition."""

import abc
import argparse
import enum
import json
import pathlib
import subprocess
import time
import typing

import numpy as np
import pandas as pd
from unittest.mock import MagicMock as ObjectDetector  # Anonimized import

import utils


DETECTION_VALUES = ["path", "face_idx", "image_width", "image_height", "time (sec)"]


class DetectorFactory:
    """Factory for loading face detection algorithms."""

    class Detector(abc.ABC):
        """Abstract class for defining face detection algorithms."""

        @staticmethod
        @abc.abstractmethod
        def load(*args, **kwargs) -> ObjectDetector:  # noqa: ANN002, ANN003
            """Return a detector based on a face detection algorithm."""
            ...

    class MTCNN:
        """Class for defining MTCNN face detector loading."""

        @staticmethod
        def load(min_proportion: float = 0.02) -> ObjectDetector:
            """
            Return a detector based on a MTCNN.

            Parameters
            ----------
            min_proportion: float
                The minimum proportion of the image to consider when detecting faces.

            Returns
            -------
            detector: MtcnnFaceDetector
                A detector based on a MTCNN.
            """
            from unittest.mock import MagicMock as MtcnnFaceDetector  # Anonimized import
            from facedet.tf import mtcnn as facedet_mtcnn

            return MtcnnFaceDetector(
                model=facedet_mtcnn.load_model(),
                min_proportion=min_proportion,
            )

    @staticmethod
    def get(detector_name: str) -> Detector:
        """Get the loader of a face detection algorithm."""
        mtcnn = DetectorFactory.MTCNN
        return locals().get(detector_name)


class Detector(str, enum.Enum):
    """Detection algorithms for face images."""

    MTCNN = "mtcnn"


def detect_faces_entrypoint(
    base_directory: str,
    detector_name: str,
    detector_config: dict[str, typing.Any] | None = None,
) -> None:
    """
    Detect faces in images. It may be used as the entry point for a
    SageMaker Processing job.

    Note that dump_empty_detections functionality is not implemented.

    Parameters
    ----------
    base_directory: str
        The base directory inside the container where the detections will be saved.
    detector_name: str
        The name of the detection algorithm to use.
    detector_config: typing.Dict[str, typing.Any]
        The configuration of the detector.

    Returns
    -------
    None
    """
    base_directory_path = pathlib.Path(base_directory)
    output_directory_path = base_directory_path / "output"
    output_directory_path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["sudo","chmod","-R","777", output_directory_path])

    detections = _detect_faces_in_dataset(
        detector=_get_detector(detector_name, detector_config),
        image_data=utils.read_face_image_data(base_directory_path / "input"),
    )
    detections = pd.concat(detections, ignore_index=True)
    detections = detections.set_index(DETECTION_VALUES)
    detections.to_csv(output_directory_path / "detections.csv")


def _get_detector(
    detector_name: str,
    detector_config: str,
) -> ObjectDetector:
    """
    Get the loader of a face detection algorithm.

    Parameters
    ----------
    detector_name: str
        The name of the detection algorithm to use.
    detector_config: str
        The configuration of the detector.

    Returns
    -------
    ObjectDetector
        The loader of the face detection algorithm.
    """
    detector_config = json.loads(detector_config) if detector_config else None
    detector = Detector(detector_name).value
    detector_loader = DetectorFactory.get(detector).load
    return detector_loader(**detector_config) if detector_config else detector_loader()


def _detect_faces_in_dataset(
    detector: ObjectDetector,
    image_data: utils.FaceImageData,
) -> typing.Generator[pd.DataFrame, None, None]:
    """
    Apply the face detection algorithm to the validation data.

    Parameters
    ----------
    detector: ObjectDetector
        The face detection algorithm to use.
    image_data: typing.Tuple[typing.Tuple[pathlib.Path,typing.IO[bytes],]
        The validation data to apply the face detection algorithm.

    Returns
    -------
    typing.Generator[pd.DataFrame, None, None]
        A generator with the face detections for each image in the validation data.
    """
    for name_rgb in image_data:

        if (detection := _detect_faces_in_an_image(detector, name_rgb)) is not None:
            yield detection


@utils.manage_exceptions()
def _detect_faces_in_an_image(
    detector: any,
    name_rgb: tuple[str, np.ndarray],
) -> pd.DataFrame:
    """
    Detect faces in an image using a face detection algorithm.

    Parameters
    ----------
    detector: any
        The face detection algorithm to use.
    name_rgb: typing.Tuple[str,np.ndarray]
        A tuple with the file name and the image data as a numpy array.

    Returns
    -------
    detection: dict
        The face detections for the image.
    """
    name, rgb_data = name_rgb
    t0, detection, t1 = time.time(), detector(rgb_data), time.time()

    # Anonimized data
    detection["path"] = ...
    detection["face_idx"] = ...
    detection["image_width"] = ...
    detection["image_height"] = ...
    detection["time (sec)"] = t1 - t0

    return detection


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detect faces in images.")
    parser.add_argument(
        "--detector_name",
        type=str,
        required=True,
        help="The name of the detection algorithm to use.",
    )
    parser.add_argument(
        "--detector_config",
        type=str,
        required=False,
        help="The configuration of the detector.",
    )
    args, _ = parser.parse_known_args()

    detect_faces_entrypoint(
        base_directory="/opt/ml/processing/",
        detector_name=args.detector_name,
        detector_config=args.detector_config,
    )
