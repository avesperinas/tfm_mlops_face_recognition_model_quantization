"""Evaluation script for pairs of images using a normalizing recipe."""

import argparse
import subprocess
import typing
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import MagicMock as IterableIterator  # Anonimized import


PAIRS_COLUMNS = ["fold", "path_1", "path_2", "tag", "distance", "exc_1", "exc_2"]
PairsMetric = typing.Generator[tuple[int, str, str, int], None, None]
InferencesMetric = typing.Generator[tuple[float, int, int], None, None]
EmbeddingMetric = typing.Generator[
    tuple[typing.Any, typing.Literal[0, 1], typing.Literal[0, 1]],
    None,
    None,
]


def _squared_l2(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute the squared L2 distance between two vectors.

    Parameters:
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns:
    -------
    np.ndarray
        Squared L2 distance between the two vectors.
    """
    return np.sum((v1 - v2) ** 2.0, axis=1)


def pairs_evaluation_entrypoint(
    base_directory: str,
    model_name: str,
) -> None:
    """
    Entry point for the pairs evaluation script.

    Parameters:
    ----------
    base_directory : Path
        The base directory inside the container where the detections will be saved.
    model_name : str
        The name of the normalization algorithm to use.

    Returns:
    -------
    None
    """
    base_directory_path = Path(base_directory)
    output_directory_path = base_directory_path / "output"
    output_directory_path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["sudo","chmod","-R","777", output_directory_path])

    pairs_evaluation_df = _evaluate_pairs(
        pairs_path=base_directory_path / "input" / "pairs.txt",
        inferences_path=base_directory_path / "processed" / f"predictions_{model_name}.csv",
    )
    pairs_evaluation_df.columns = PAIRS_COLUMNS
    pairs_evaluation_df.to_csv(
        output_directory_path / f"pairs_evaluation_{model_name}.csv",
        index=False
    )


def _evaluate_pairs(
    pairs_path: Path,
    inferences_path: Path,
) -> pd.DataFrame:
    """
    Evaluate pairs of images using the inferences.

    Parameters:
    ----------
    pairs_path : Path
        Path to the pairs CSV file.
    inferences_path : Path
        Path to the inferences CSV file.

    Returns:
    -------
    pairs_evaluation_df: pd.DataFrame
        DataFrame containing the evaluation results.
    """
    pairs = _read_pairs(pairs_path)
    metrics_function = _make_inferences_metrics_function(inferences_path)

    pairs_evaluation = _evaluate_inferences(pairs, metrics_function)
    pairs_evaluation_df = pd.DataFrame(pairs_evaluation)

    return pairs_evaluation_df


def _read_pairs(
    pairs_path: Path,
) -> PairsMetric:
    """
    Read pairs from a file.

    The implementation needs to read twice the pairs file, avoiding
    the overhead of storing into memory all the pairs combinations
    (they can be several million comparisons).

    Parameters:
    ----------
    pairs_path : Path
        Path to the pairs file.
    Returns:
    -------
    PairsMetric
        A generator yielding tuples of (fold, file_1, file_2, tag).
    """
    yield ...


def _make_inferences_metrics_function(
    inferences_path: str,
    metric: typing.Callable[[np.ndarray, np.ndarray], np.ndarray] = _squared_l2,
    default: float = float("inf"),
) -> typing.Callable[[typing.Iterator[tuple[int, int]]], EmbeddingMetric]:
    """
    Create a batch metric function for embeddings.

    Parameters:
    ----------
    embeddings : pd.DataFrame
        DataFrame containing the embeddings.
    metric : typing.Callable
        Function to compute the distance between two embeddings.
    default : float
        Default value to return when an embedding is not found.

    Returns:
    -------
    typing.Callable
        A function that takes a batch of pairs and returns the computed metric.
    """
    return ...


def _evaluate_inferences(
    pairs: PairsMetric,
    metrics: InferencesMetric,
    batch_size: int = 1024,
) -> typing.Generator[tuple[int, str, str, int, float, int, int], None, None]:
    """
    Evaluate pairs of images using the inferences.

    Parameters:
    ----------
    pairs : typing.Generator
        Pairs of images to evaluate.
    metrics : typing.Generator
        Metrics to use for evaluation.
    batch_size : int
        Size of the batch for evaluation.


    Returns:
    -------
    typing.Generator
        A generator yielding tuples of DATA.
    """
    for items in IterableIterator(pairs, batch_size):
        batch = [(path_1, path_2) for _, path_1, path_2, _ in items]
        for row, data in zip(items, metrics(batch), strict=False):
            yield row, data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Normalize faces in a dataset.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the normalization algorithm to use.",
    )
    args, _ = parser.parse_known_args()

    pairs_evaluation_entrypoint(
        base_directory="/opt/ml/processing",
        model_name=args.model_name,
    )
