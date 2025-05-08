"""Buffer utilities for image processing."""

import enum
import functools
import io
import itertools
import typing

import numpy as np

from .exception_management import manage_exceptions


class ProcessFormat(str, enum.Enum):
    """Class to define the process format of the image."""

    JPEG = "JPEG"
    PNG = "PNG"
    TIFF = "TIFF"
    NPZ = "NPZ"
    NPY = "NPY"


EXTENSIONS_MAPPING = {
    ProcessFormat.JPEG: (".jpeg", ".jpg"),
    ProcessFormat.PNG: (".png",),
    ProcessFormat.TIFF: (".tiff",),
    ProcessFormat.NPZ: (".npz",),
    ProcessFormat.NPY: (".npy",),
}


@manage_exceptions()
def write_array_into_buffer(array: np.ndarray, compress: bool) -> bytes:
    """
    Write a numpy array into a buffer.

    Parameters:
    ----------
    array: np.ndarray
        The numpy array to write into the buffer.
    compress: bool
        Whether to compress the buffer or not.

    Returns:
    -------
    buffer: buf
        The buffer with the numpy array.
    """
    import zlib

    f_hdl = io.BytesIO()
    np.save(f_hdl, array)
    buf = f_hdl.getvalue()
    if compress:
        buf = zlib.compress(buf)
    return buf


@manage_exceptions()
def write_image_into_buffer(rgb_image: np.ndarray, ext: str) -> bytes:
    """
    Write an image into a buffer.

    Parameters:
    ----------
    rgb_image: np.ndarray
        The image to write into the buffer.
    ext: str
        The extension of the image.

    Returns:
    -------
    buffer: buf
        The buffer with the image.
    """
    import cv2

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    _, arr = cv2.imencode(ext, bgr_image)
    return arr.tobytes()


def make_array_to_buffer(
    allowed_extensions: tuple[str] | list[str] | set[str],
) -> typing.Callable[[np.ndarray, str], bytes]:
    """Create a function that maps an array to a buffer.

    Parameters:
    ----------
    allowed_extensions: tuple[str] | list[str] | set[str]
        The allowed extensions for the array.

    Returns:
    -------
    array_to_buffer_mapping: typing.Callable[[np.ndarray, str], bytes]
        The function that maps an array to a buffer.
    """
    array_to_buffer_map = {
        ".npz": functools.partial(write_array_into_buffer, compress=True),
        ".npy": functools.partial(write_array_into_buffer, compress=False),
        ".jpeg": functools.partial(write_image_into_buffer, ext=".jpeg"),
        ".jpg": functools.partial(write_image_into_buffer, ext=".jpg"),
        ".png": functools.partial(write_image_into_buffer, ext=".png"),
        ".tiff": functools.partial(write_image_into_buffer, ext=".tiff"),
    }

    def array_to_buffer_mapping(
        array: list[str],
        ext: str,
        mapper: dict[str, typing.Any],
    ) -> bytes:
        """
        Map an array to a buffer.

        Parameters:
        ----------
        array: list[str]
            The array to map to a buffer.
        ext: str
            The extension of the array.
        mapper: dict[str, typing.Any]
            The mapper to use for the array.

        Returns:
        -------
        bytes
            The buffer with the array.
        """
        return mapper[ext.lower()](array)

    filtered_mapper = {k: v for k, v in array_to_buffer_map.items() if k in allowed_extensions}
    return functools.partial(array_to_buffer_mapping, mapper=filtered_mapper)


def array_to_buffer(array: np.ndarray, suffix: str) -> bytes:
    """
    Convert an array to a buffer.

    Parameters:
    ----------
    array: np.ndarray
        The array to convert to a buffer.
    suffix: str
        The suffix of the array.

    Returns:
    -------
    bytes
        The buffer with the array.
    """
    process_extensions = set(
        itertools.chain(
            *[
                EXTENSIONS_MAPPING[ProcessFormat[x.name]]
                for x in (ProcessFormat.JPEG, ProcessFormat.PNG, ProcessFormat.TIFF)
            ],
        ),
    )
    return make_array_to_buffer(process_extensions)(array, suffix)
