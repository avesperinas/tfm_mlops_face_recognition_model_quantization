"""Face normalization step definition."""

import argparse
import enum
import functools
import io
import json
import subprocess
import tarfile
import typing
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import MagicMock as types,  MagicMock as object_detectors  # Anonimized import

import utils


PROCESSED_EXTENSIONS = {".jpeg", ".jpg", ".png", ".tiff"}
FaceNormalizer = typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
CriterionMarker = typing.Callable[[np.ndarray], np.ndarray]
NormalizedFace = tuple[io.BytesIO, str]


class SelectionCriterion(str, enum.Enum):
    """Criteria to select faces from an image."""

    ALL_FACES = "all"
    LARGEST_FACE = "largest"
    MOST_CENTERED_AND_LARGEST_FACE = "most-centered-and-largest"
    MOST_CENTERED_FACE = "most-centered"


class Normalizer(str, enum.Enum):
    """Normalization algorithms for face images."""

    SIMTRANS = "simtrans"


class NormalizerFactory:
    """Factory class for face normalization algorithms."""

    class SIMTRANS:
        """Factory class for SIMTRANS face normalization algorithm."""

        @staticmethod
        def load(
            object_shape: tuple[str, str] | None = (112, 112),
            margin: tuple[int, int] | None = (0, 0),
        ) -> FaceNormalizer:
            """
            Load SIMTRANS face normalization algorithm.

            Parameters
            ----------
            object_shape: typing.Tuple[str,str]
                Shape of the normalized face image.
            margin: typing.Tuple[int, int]
                Margin size added around normalized faces.

            Returns
            -------
            FaceNormalizer
                SIMTRANS face normalization algorithm.
            """
            from unittest.mock import MagicMock as algorithms  # Anonimized import

            return lambda _: functools.partial(
                ...
            )


def normalize_faces_entrypoint(
    base_directory: str,
    normalizer_name: str,
    criteria_name: str,
    normalizer_config: dict[str, typing.Any] | None = None,
) -> None:
    """
    Normalize faces in a dataset. It may be used as the entry point for a
    SageMaker Processing job.

    Parameters
    ----------
    base_directory: str
        The base directory inside the container where the detections will be saved.
    normalizer_name: str
        The name of the normalization algorithm to use.
    criteria_marker: str
        The criterion to select faces from an image.
    normalizer_config: typing.Optional[typing.Dict[str,typing.Any]]
        The configuration of the normalizer.

    Returns
    -------
    None
    """
    base_directory_path = Path(base_directory)
    output_directory_path = base_directory_path / "output"
    output_directory_path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["sudo","chmod","-R","777", output_directory_path])

    normalized_faces = _normalize_faces_in_dataset(
        images_data=utils.read_face_image_data(base_directory_path / "input"),
        detections=_read_detections(base_directory_path / "processed"),
        normalizer=_get_normalizer(normalizer_name, normalizer_config),
        criteria_marker=_get_criteria_marker(SelectionCriterion(criteria_name)),
    )

    with tarfile.open(output_directory_path / "normalized_faces.tar", "w") as tar:
        for normalized_face in normalized_faces:

            normalized_face_buf, normalized_face_path = normalized_face
            tarinfo = tarfile.TarInfo(name=normalized_face_path)
            tarinfo.size = len(normalized_face_buf.getvalue())
            normalized_face_buf.seek(0)
            tar.addfile(tarinfo, normalized_face_buf)


def _read_detections(
    base_directory_path: Path,
) -> pd.DataFrame:
    """
    Read detections from a CSV file, filtering out the unnamed columns.

    Parameters
    ----------
    base_directory_path: Path
        The base directory inside the container where the detections will be saved.

    Returns
    -------
    pd.DataFrame
        The detections read from the CSV file.
    """
    detections = pd.read_csv(base_directory_path / "detections.csv", header=0)
    detections.set_index("path", inplace=True)  # noqa: PD002
    detections.columns = detections.columns.astype(str)
    return detections.loc[:, ~detections.columns.str.contains("^Unnamed")]


def _get_normalizer(
    normalizer_name: str,
    normalizer_config: str,
) -> FaceNormalizer:
    """
    Get the face normalization algorithm.

    Parameters
    ----------
    normalizer_name: str
        The name of the normalization algorithm to use.
    normalizer_config: str
        The configuration of the normalizer.

    Returns
    -------
    FaceNormalizer
        The face normalization algorithm.
    """
    normalizer_config = json.loads(normalizer_config) if normalizer_config else None
    normalizer_loader = _get_normalizer_loader(Normalizer(normalizer_name))
    return normalizer_loader(**normalizer_config) if normalizer_config else normalizer_loader()


def _get_normalizer_loader(
    normalizer_name: str,
) -> FaceNormalizer:
    """
    Get the loader of a face normalization algorithm.

    Parameters
    ----------
    normalizer_name: str
        The name of the normalization algorithm to use.

    Returns
    -------
    FaceNormalizer
        The loader of the face normalization algorithm.
    """
    normalizer_factories = {
        Normalizer.SIMTRANS: NormalizerFactory.SIMTRANS.load,
    }
    return normalizer_factories.get(normalizer_name)


def _get_criteria_marker(criteria_name: SelectionCriterion) -> CriterionMarker:
    """
    Get the criterion marker for selecting faces from an image.

    Parameters
    ----------
    criteria_name: SelectionCriterion
        The criterion to select faces from an image.

    Returns
    -------
    SelectionCriterion
        The criterion marker for selecting faces from an image
    """
    from unittest.mock import MagicMock as selection_criteria  # Anonimized import

    criteria_factories = {
        ...
    }
    return criteria_factories.get(criteria_name)


def _normalize_faces_in_dataset(
    images_data: utils.FaceImageData,
    detections: pd.DataFrame,
    normalizer: FaceNormalizer,
    criteria_marker: CriterionMarker,
) -> typing.Generator[NormalizedFace, None, None]:
    """
    Normalize faces in a dataset.

    Parameters
    ----------
    images_data: utils.FaceImageData
        The images to normalize.
    detections: pd.DataFrame
        The face detections.
    normalizer: FaceNormalizer
        The face normalization algorithm.
    criteria_marker: CriterionMarker
        The criterion marker for selecting faces from an image.

    Returns
    -------
    typing.Generator[NormalizedFace, None, None]
        A generator with the normalized faces.
    """
    for image_data in images_data:
        yield from _normalize_one_image(
            image_data=image_data,
            detections=detections,
            normalizer_maker=normalizer,
            criterion_maker=criteria_marker,
        )


def _normalize_one_image(
    image_data: utils.FaceImageData,
    detections: pd.DataFrame,
    normalizer_maker: FaceNormalizer,
    criterion_maker: CriterionMarker,
) -> typing.Generator[NormalizedFace, None, None]:
    """
    Normalize all faces from an image.

    Parameters
    ----------
    image_data: utils.FaceImageData
        The image to normalize.
    detections: pd.DataFrame
        The face detections.
    normalizer_maker: FaceNormalizer
        The face normalization algorithm.
    criterion_maker: CriterionMarker
        The criterion marker for selecting faces from an image.

    Returns
    -------
    typing.Generator[NormalizedFace, None, None]
        A generator with the normalized faces.
    """
    path_str, path, image = image_data[0], Path(image_data[0]), image_data[1]

    image_detections = ...
    raw_bboxes, bboxes, keypoints = ...

    images_iterable = zip(
        image_detections.iterrows(),
        keypoints,
        bboxes.iterrows(),
        strict=False,
    )

    for detection, keypoint, b_box in images_iterable:

        face_data = normalizer_maker(b_box)(image, keypoint)
        idx = int(detection[1]["face_idx"])

        buf = utils.array_to_buffer(face_data, path.suffix)
        new_path_str = f"{path.stem}_{idx}{path.suffix}"
        yield io.BytesIO(buf), new_path_str


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Normalize faces in a dataset.")
    parser.add_argument(
        "--normalizer_name",
        type=str,
        required=True,
        help="The name of the normalization algorithm to use.",
    )
    parser.add_argument(
        "--normalizer_config",
        type=dict[str, typing.Any],
        required=False,
        help="The configuration of the normalizer.",
    )
    parser.add_argument(
        "--criteria_name",
        type=str,
        required=True,
        help="The criterion to select faces from an image.",
    )
    args, _ = parser.parse_known_args()

    normalize_faces_entrypoint(
        base_directory="/opt/ml/processing/",
        normalizer_name=args.normalizer_name,
        normalizer_config=args.normalizer_config,
        criteria_name=args.criteria_name,
    )
