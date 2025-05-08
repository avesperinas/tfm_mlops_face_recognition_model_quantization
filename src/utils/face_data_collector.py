"""Module to define data collectors for face data gathering."""

import pathlib
import tarfile
import typing

from .data_loaders import load_file


FaceImageData = tuple[
    tuple[
        pathlib.Path,
        typing.IO[bytes],
    ]
]


def read_face_image_data(
    source_data_path: pathlib.Path,
) -> typing.Generator[FaceImageData, None, None]:
    """
    Read the validation data from the base directory, filtering only
    the image files and returning a tuple with the file path and the image data.
    It returns only the not null image data.

    Parameters
    ----------
    source_data_path: pathlib.Path
        The base directory inside the container where the detections will be saved.

    Returns
    -------
    typing.Tuple[typing.Tuple[pathlib.Path,typing.IO[bytes],]
        A tuple with the file paths and the images data for each valid image file.
    """
    return (
        data_pair
        for tar_path in source_data_path.rglob("*.tar")
        for data_pair in load_data_from_tar_file(tar_path)
    )


def load_data_from_tar_file(
    tar_path: pathlib.Path,
    extensions: tuple[str] | None = (".jpg", ".jpeg", ".png"),
) -> typing.Generator[tuple[pathlib.Path], None, None]:
    """
    Load the data from a tar file.

    Parameters
    ----------
    tar_path: pathlib.Path
        The path to the tar file.
    extensions: typing.Tuple[str]
        The file extensions to filter.

    Returns
    -------
    typing.Tuple[pathlib.Path]
        A tuple with the file paths.
    """
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():

            if member.isfile() and member.name.endswith(extensions):
                file_data = load_file(
                    file_path=pathlib.Path(member.name),
                    file_obj=tar.extractfile(member),
                )

                if file_data:
                    yield file_data
