import pickle

import numpy as np
import pandas as pd


def zero_aware_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return metrics that expose errors hidden by the benchmark score."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    mae = float(np.mean(np.abs(actual - predicted))) if actual.size else 0.0
    total_actual = float(np.abs(actual).sum())
    wmape = float(np.abs(actual - predicted).sum() / total_actual) if total_actual else 0.0
    actual_positive = actual > 0
    predicted_positive = predicted > 0
    true_positive = int(np.sum(actual_positive & predicted_positive))
    predicted_count = int(np.sum(predicted_positive))
    actual_count = int(np.sum(actual_positive))
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / actual_count if actual_count else 0.0
    return {
        "mae": mae,
        "wmape": wmape,
        "positive_precision": precision,
        "positive_recall": recall,
        "predicted_positive_rate": float(np.mean(predicted_positive)) if actual.size else 0.0,
    }


def evaluate(submission_path, answer_path):

    # =====================================================
    # LOAD
    # =====================================================

    with open(submission_path, "rb") as f:
        submission = pickle.load(f)

    with open(answer_path, "rb") as f:
        answer = pickle.load(f)

    # =====================================================
    # CONVERT TO DATAFRAME
    # =====================================================

    submission_df = pd.DataFrame(submission)

    answer_df = pd.DataFrame(answer)

    submission_df["item_id"] = (
        submission_df["item_id"]
        .astype(str)
    )

    answer_df["item_id"] = (
        answer_df["item_id"]
        .astype(str)
    )

    # =====================================================
    # MERGE
    # =====================================================

    merged_df = answer_df.merge(
        submission_df,
        on=["location", "item_id"],
        how="left",
        suffixes=("_gt", "_pred")
    )

    # missing prediction -> 0
    merged_df["quantity_pred"] = (
        merged_df["quantity_pred"]
        .fillna(0)
    )

    # =====================================================
    # REMOVE ZERO GT
    # =====================================================

    merged_df = merged_df[
        merged_df["quantity_gt"] != 0
    ]

    if len(merged_df) == 0:
        return 0.0

    # =====================================================
    # MAPE
    # =====================================================

    ape = (
        np.abs(
            merged_df["quantity_gt"]
            - merged_df["quantity_pred"]
        )
        /
        np.abs(
            merged_df["quantity_gt"]
        )
    )

    mape = 100.0 - ape.mean() * 100

    return round(mape, 4)
