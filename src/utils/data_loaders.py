"""Module to implement the file loaders."""

import pathlib
import typing

import numpy as np

from .exception_management import manage_exceptions


LOADER_MAP = {
    ".jpeg": lambda file_path: _load_image(file_path),
    ".jpg": lambda file_path: _load_image(file_path),
    ".png": lambda file_path: _load_image(file_path),
    ".tiff": lambda file_path: _load_image(file_path),
}


def load_file(
    file_path: pathlib.Path,
    file_obj: typing.IO | None = None,
) -> tuple[str, np.ndarray]:
    """
    Load a file from a file path. The function will return the file
    path and the file data as a numpy ndarray.

    Parameters
    ----------
    file_path: pathlib.Path
        The file path to the image file.
    file_obj: typing.IO
        The info of the image file.

    Returns
    -------
    typing.Tuple[str,np.ndarray]
        A tuple with the file path and the image data as a numpy array.
    """
    loader_function = LOADER_MAP.get(file_path.suffix)
    return str(file_path), loader_function(file_obj) if loader_function else None


@manage_exceptions()
def _load_image(
    file_obj: typing.IO,
) -> tuple[str, np.ndarray]:
    """
    Load an image from a file path. The function will return the file
    path and the image data as a numpy ndarray.

    Parameters
    ----------
    file_obj: typing.IO
        The info of the image file.

    Returns
    -------
    np.ndarray
        The image data as a numpy array.
    """
    from loaders import load_rgb_image

    rgb_img: np.ndarray = load_rgb_image(
        file_obj,
        reject_nonstd_exif=False,
    )
    return rgb_img
