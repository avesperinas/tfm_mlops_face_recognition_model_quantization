"""Generates the report for the given evaluation."""

import typing
from functools import reduce

import numpy as np
import pandas as pd
from sklearn import metrics


MIN_DIRECTION = "min"
RocCurve = tuple[np.ndarray, np.ndarray, np.ndarray]


def generate_roc_curve(
    evaluations: pd.DataFrame,
    true_name: str,
    score_name: str,
) -> RocCurve:
    """
    Generate the ROC curve from the given evaluations.

    Parameters:
    ----------
    evaluations : pd.DataFrame
        DataFrame containing the evaluation data.
    true_name : str
        Name of the column containing the true labels.
    score_name : str
        Name of the column containing the scores.

    Returns:
    -------
    fprs : np.ndarray
        False positive rates.
    tprs : np.ndarray
        True positive rates.
    ths : np.ndarray
        Thresholds.
    """
    y_true = evaluations[true_name]
    y_score = evaluations[score_name]

    if np.sum(y_true == 0) == 0:
        raise RuntimeError("Data has 0 negative samples")  # noqa: TRY003, EM101

    if np.sum(y_true == 1) == 0:
        raise RuntimeError("Data has 0 positive samples")  # noqa: TRY003, EM101

    return metrics.roc_curve(y_true, y_score, drop_intermediate=False)


def normalize_roc(
    roc_curve: RocCurve,
    direction: str,
) -> RocCurve:
    """
    Normalize the ROC curve by inverting the false positive rates
    if the direction is 'min'.

    Parameters:
    ----------
    roc_curve : RocCurve
        The ROC curve to normalize.
    direction : str
        The direction of the metric.

    Returns:
    -------
    roc_curve : RocCurve
        The normalized ROC curve.
    """
    if direction == MIN_DIRECTION:
        roc_curve = roc_curve[:2] + (-roc_curve[2],)
    return roc_curve


def normalize_score(
    evaluations: pd.DataFrame,
    score_name: str,
    direction: str,
) -> pd.DataFrame:
    """
    Normalize the score in the evaluations DataFrame.

    Parameters:
    ----------
    evaluations : pd.DataFrame
        DataFrame containing the evaluation data.
    score_name : str
        Name of the column containing the scores.
    direction : str
        The direction of the metric.

    Returns:
    -------
    evaluations : pd.DataFrame
        DataFrame containing the normalized scores.
    """
    threshold = 999

    if direction == MIN_DIRECTION:
        evaluations = evaluations.copy()
        evaluations.loc[:, score_name] = -evaluations[score_name]
        evaluations.loc[evaluations[score_name] < -threshold, score_name] = -threshold

    return evaluations


def generate_working_points_with_uncertainty(
    fprs: np.ndarray,
    tprs: np.ndarray,
    ths: np.ndarray,
    n_positives: int,
    n_negatives: int,
) -> dict[str, np.ndarray | list[str]]:
    """
    Generate working points with uncertainty.

    Parameters:
    ----------
    fprs : np.ndarray
        False positive rates.
    tprs : np.ndarray
        True positive rates.
    ths : np.ndarray
        Thresholds.
    n_positives : int
        Number of positive samples.
    n_negatives : int
        Number of negative samples.

    Returns:
    -------
    working_points : dict
        Dictionary containing the working points with uncertainty.
    """
    hters = _compute_hter(fprs, 1 - tprs)
    fpr_points = _scope_fpr(fprs)
    indexes = [np.argmin(fprs < x) for x in fpr_points]
    uncertainties = [
        _estimate_ci_95_for_paired_proportions(
            tpr,
            fpr,
            n_positives,
            n_negatives,
        )
        for tpr, fpr in zip(tprs[indexes], fprs[indexes], strict=False)
    ]

    return {
        "fpr": fprs[indexes],
        "tpr": tprs[indexes],
        "uncertainty": uncertainties,
        "hter": hters[indexes],
        "th": ths[indexes],
    }


def count_exceptions(
    evaluations: pd.DataFrame,
    exceptions: list[typing.Any],
) -> np.ndarray:
    """
    Count the number of exceptions in the evaluations DataFrame.

    Parameters:
    ----------
    evaluations : pd.DataFrame
        DataFrame containing the evaluation data.
    exceptions : list
        List of exceptions to count.

    Returns:
    -------
    np.ndarray
        Number of exceptions.
    """
    failures = reduce(
        lambda x, y: x | y,
        [evaluations[column] == 1 for column in exceptions],
        0,
    )
    return np.sum(failures)


def generate_metrics(
    fprs: np.ndarray,
    tprs: np.ndarray,
    ths: np.ndarray,
    num_negative: int,
    num_positive: int,
    num_exceptions: int = 0,
) -> typing.Any:  # noqa: ANN401
    """
    Generate the metrics from the given ROC curve.

    Parameters:
    ----------
    fprs : np.ndarray
        False positive rates.
    tprs : np.ndarray
        True positive rates.
    ths : np.ndarray
        Thresholds.
    num_negative : int
        Number of negative samples.
    num_positive : int
        Number of positive samples.
    num_exceptions : int
        Number of exceptions.

    Returns:
    -------
    metrics : dict
        Dictionary containing the metrics.
    """
    total = num_negative + num_positive
    auc = metrics.auc(fprs, tprs)
    fnrs = 1.0 - tprs
    eer_idx = _compute_eer(fprs, fnrs)
    hters = _compute_hter(fprs, fnrs)
    hter_idx = np.argmin(hters)
    accs = 1.0 - _compute_hter(
        fprs,
        fnrs,
        negative_weight=num_negative,
        positive_weight=num_positive,
    )
    acc_idx = np.argmax(accs)
    exc_rate = num_exceptions / float(total)
    return {
        "acc": [accs[acc_idx]],
        "acc_th": [ths[acc_idx]],
        "eer": [fprs[eer_idx]],
        "eer_th": [ths[eer_idx]],
        "hter": [hters[hter_idx]],
        "hter_th": [ths[hter_idx]],
        "auc": [auc],
        "exc": [exc_rate],
    }


def filter_fte(
    eval_with_fte: pd.DataFrame,
    exceptions: typing.List[typing.Any]
) -> pd.DataFrame:
    """
    Filter the evaluation DataFrame to remove exceptions.

    Parameters:
    ----------
    eval_with_fte : pd.DataFrame
        DataFrame containing the evaluation data.
    exceptions : list
        List of exceptions to filter out.

    Returns:
    -------
    eval_with_fte : pd.DataFrame
        DataFrame containing the filtered evaluation data.
    """
    non_failures = reduce(
        lambda x, y: x & y,  # type: ignore
        [(eval_with_fte[x] != 1) for x in exceptions],
        [True]*eval_with_fte.shape[0],
    )
    return eval_with_fte.loc[non_failures, :]


def _compute_hter(
    fprs: np.ndarray,
    fnrs: np.ndarray,
    negative_weight: float = 0.5,
    positive_weight: float = 0.5,
) -> np.ndarray:

    total = negative_weight + positive_weight
    return ((negative_weight * fprs) + (positive_weight * fnrs)) / float(total)


def _compute_eer(fprs: np.ndarray, fnrs: np.ndarray) -> np.ndarray:

    return np.argmin(np.abs(fprs - fnrs))


def _lower_bound_magnitude_order(value: float) -> float:

    order = 1.0
    while order > value:
        order *= 0.1
    return order


def _scope_fpr(fprs: np.ndarray) -> list[float]:

    lower_bound_fpr = _lower_bound_magnitude_order(
        np.min(fprs[fprs > 0.0]),
    )
    return [0.1, 0.05, 0.01] + [10**x for x in range(-3, int(np.log10(lower_bound_fpr)), -1)]


def _estimate_ci_95_for_paired_proportions(
    proportion_1: np.array,
    proportion_2: np.array,
    sample_size_1: np.array,
    sample_size_2: np.array,
    averaging: bool = False,
) -> np.array:

    c = 4 * averaging + 1 * (not averaging)
    return 1.96 * np.sqrt(
        1 / c * proportion_1 * (1 - proportion_1) / sample_size_1
        + 1 / c * proportion_2 * (1 - proportion_2) / sample_size_2,
    )
