"""Module to evaluate pairs of images using the inferences."""

import argparse
import subprocess
import typing
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from utils import generate_report


FOLD_NAME = "fold"
TAG_NAME = "tag"
COLUMN_NAME = "distance"
DIRECTION = "min"


def verification_metrics_entrypoint(
    base_directory: str,
    model_name: str,
) -> None:
    """
    Entry point for the verification metrics script.

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
    output_file_path = output_directory_path / f"verification_metrics_{model_name}.csv"
    output_roc_path = output_directory_path / f"roc_{model_name}.csv"
    output_directory_path.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["sudo","chmod","-R","777", output_directory_path])

    verification_metrics, roc_data = _get_verification_metrics(
        pairs_evaluation_path=base_directory_path / "processed" / f"pairs_evaluation_{model_name}.csv",
    )
    verification_metrics.to_csv(output_file_path, index=False)
    roc_data.to_csv(output_roc_path, index=False)


def _get_verification_metrics(
    pairs_evaluation_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the verification metrics from the pairs evaluation.

    Parameters:
    ----------
    pairs_evaluation_path : Path
        Path to the pairs evaluation CSV file.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing the verification metrics.
    pd.DataFrame
        DataFrame containing the ROC data.
    """
    pairs_evaluation_df = pd.read_csv(pairs_evaluation_path)
    evaluation, exceptions = _trim_pairs_evaluation(pairs_evaluation_df)
    return _compute_verification_metrics(evaluation, exceptions)


def _trim_pairs_evaluation(
    pairs_evaluation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Salvage the pairs evaluation DataFrame.

    Parameters:
    ----------
    pairs_evaluation_df : pd.DataFrame
        DataFrame containing the pairs evaluation.

    Returns:
    -------
    evaluation: pd.DataFrame
        DataFrame containing the trimmed pairs evaluation.
    exceptions: pd.DataFrame
        DataFrame containing the exceptions.
    """

    def _make_usecols_for_evaluation(
            
    ) -> typing.Callable[[str], bool]: return ...

    def _trim_dataframe(
        dataframe: pd.DataFrame,
        usecols: typing.Callable[[str], bool],
    ) -> pd.DataFrame: return ...

    def _get_evaluation_exceptions(
            dataframe: pd.DataFrame
        ) -> pd.DataFrame: return ...

    def _check_fold_column(
            dataframe: pd.DataFrame
        ) -> pd.DataFrame: return ...

    evaluation = _trim_dataframe(
        pairs_evaluation_df,
        usecols=_make_usecols_for_evaluation(),
    )
    exceptions = _get_evaluation_exceptions(evaluation)
    evaluation = _check_fold_column(evaluation)

    return evaluation, exceptions


def _compute_verification_metrics(
    evaluation: pd.DataFrame,
    exceptions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the verification metrics.

    Parameters:
    ----------
    evaluation : pd.DataFrame
        DataFrame containing the evaluation data.
    exceptions : pd.DataFrame
        DataFrame containing the exceptions.

    Returns:
    -------
    metrics: pd.DataFrame
        DataFrame containing the verification metrics.
    roc_data: pd.DataFrame
        DataFrame containing the ROC data.
    """

    def _compute_metrics(
        evaluation: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        generate_roc_curve = partial(
            generate_report.generate_roc_curve,
            true_name=TAG_NAME,
            score_name=COLUMN_NAME,
        )
        raw_roc_curve = generate_roc_curve(
            generate_report.normalize_score(
                evaluations=evaluation,
                score_name=COLUMN_NAME,
                direction=DIRECTION,
            ),
        )
        roc_curve = generate_report.normalize_roc(
            roc_curve=raw_roc_curve,
            direction=DIRECTION,
        )
        fprs, tprs, ths = roc_curve
        roc_data = pd.DataFrame({
            'False Positive Rate': fprs,
            'True Positive Rate': tprs,
            'Threshold': ths
        })
        working_points = pd.DataFrame(
            data=generate_report.generate_working_points_with_uncertainty(
                *roc_curve,
                n_positives=len(evaluation[evaluation[TAG_NAME] == 1]),
                n_negatives=len(evaluation[evaluation[TAG_NAME] == 0]),
            ),
        ).drop("hter", axis=1)

        metrics = pd.DataFrame(
            data=generate_report.generate_metrics(
                *roc_curve,
                num_negative=np.sum(1 - evaluation[TAG_NAME]),
                num_positive=np.sum(evaluation[TAG_NAME]),
                num_exceptions=generate_report.count_exceptions(
                    evaluation,
                    exceptions,
                ),
            ),
        ).drop(["hter", "hter_th", "acc_th", "eer_th"], axis=1)
        return working_points.merge(metrics, how="cross"), roc_data

    evaluation_without_fte = generate_report.filter_fte(
        evaluation,
        exceptions=exceptions,
    )

    metrics_without_fte, _ = _compute_metrics(evaluation_without_fte)
    metrics_with_fte, roc_data = _compute_metrics(evaluation)

    metrics_without_fte["FTE"] = 0
    metrics_with_fte["FTE"] = 1

    metrics = pd.concat((metrics_without_fte, metrics_with_fte))
    metrics = metrics.reset_index(drop=True)
    metrics.index.name = "index"
    return metrics, roc_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Normalize faces in a dataset.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the normalization algorithm to use.",
    )
    args, _ = parser.parse_known_args()

    verification_metrics_entrypoint(
        base_directory="/opt/ml/processing",
        model_name=args.model_name,
    )
