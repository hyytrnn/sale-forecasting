import argparse
import json
import pickle
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from . import train_predict as tp
from .project_paths import OUTPUT_DIR, ensure_output_dirs


def log(message: str) -> None:
    print(message, flush=True)


def run_id() -> str:
    return time.strftime("decval_%Y%m%d_%H%M%S")


def month_days(year: int, month: int) -> int:
    start, end = tp.month_bounds(year, month)
    return (end - start).days + 1


def candidate_pairs_for_locations(locations: pl.DataFrame, active_items: pl.DataFrame, start: int, size: int) -> pl.DataFrame:
    loc_values = locations["location"].to_list()
    if size:
        loc_values = loc_values[:size]
    batch_locations = pl.DataFrame({"location": loc_values[start : start + size] if size else loc_values[start:]})
    return batch_locations.join(active_items.select("item_id"), how="cross")


def predict_candidate_grid(
    args,
    bad_days: set[date],
    locations: pl.DataFrame,
    active_items: pl.DataFrame,
    model_bundle: dict,
    cutoff: date,
    target_month: int,
    target_days: int,
    output_path: Path,
) -> pd.DataFrame:
    daily_lf = tp.daily_valid_lf(bad_days)
    feature_tables = tp.build_feature_tables(daily_lf, active_items, cutoff)
    rows: list[pd.DataFrame] = []
    loc_values = locations["location"].to_list()
    if args.max_prediction_locations:
        loc_values = loc_values[: args.max_prediction_locations]

    for start in range(0, len(loc_values), args.location_batch_size):
        batch_locations = pl.DataFrame({"location": loc_values[start : start + args.location_batch_size]})
        pairs = batch_locations.join(active_items.select("item_id"), how="cross")
        features = tp.add_features_from_tables(pairs, active_items, feature_tables, cutoff, target_month, target_days)
        if args.prediction_strategy == "catboost":
            pred = tp.predict_quantity(
                args,
                model_bundle["clf"],
                model_bundle["reg"],
                model_bundle["bucket_model"],
                model_bundle["bucket_values"],
                features,
                model_bundle["feature_columns"],
                model_bundle["cat_features"],
                model_bundle["tune"]["threshold"],
                model_bundle["tune"]["scale"],
                model_bundle["tune"]["blend"],
                model_bundle["tune"]["bucket_blend"],
                model_bundle["tune"]["proba_power"],
            )
        else:
            pred = tp.apply_baseline_strategy(features, args.prediction_strategy, args.final_scale)

        out = features[["location", "item_id"]].copy()
        out["quantity"] = pred.astype(np.int64)
        rows.append(out)
        log(f"Predicted {target_month:02d}: locations {start + 1:,}-{min(start + args.location_batch_size, len(loc_values)):,} / {len(loc_values):,}")

    submission = pd.concat(rows, ignore_index=True)
    submission = submission[["location", "item_id", "quantity"]]
    submission = pd.DataFrame(
        {
            "location": submission["location"].astype(str).to_numpy(dtype=object),
            "item_id": submission["item_id"].astype(str).to_numpy(dtype=object),
            "quantity": submission["quantity"].astype(np.int64).to_numpy(),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(submission, f)
    return submission


def score_december_submission(submission: pd.DataFrame, bad_days: set[date], active_items: pl.DataFrame) -> dict:
    daily_lf = tp.daily_valid_lf(bad_days)
    actual = tp.monthly_target(daily_lf, active_items, 2025, 12, bad_days).to_pandas()
    actual["location"] = actual["location"].astype(str)
    actual["item_id"] = actual["item_id"].astype(str)
    merged = actual.merge(submission, on=["location", "item_id"], how="left", suffixes=("_actual", "_pred"))
    merged["quantity_pred"] = merged["quantity_pred"].fillna(0)
    score = tp.score_metric(merged["quantity_actual"].to_numpy(), merged["quantity_pred"].to_numpy())
    ape = np.abs(merged["quantity_actual"].to_numpy() - merged["quantity_pred"].to_numpy()) / np.abs(
        merged["quantity_actual"].to_numpy()
    )
    return {
        "december_score": score,
        "december_positive_rows": int(len(merged)),
        "december_actual_sum": float(merged["quantity_actual"].sum()),
        "december_pred_sum_on_positive_rows": float(merged["quantity_pred"].sum()),
        "december_mean_ape": float(ape.mean()),
        "december_median_ape": float(np.median(ape)),
    }


def train_bundle(args, bad_days: set[date], locations: pl.DataFrame, active_items: pl.DataFrame, end_month: int) -> dict:
    original_end_month = args.end_month
    args.end_month = end_month
    train_df = tp.build_training_data(args, bad_days, locations, active_items)
    log(f"Training rows through month {end_month:02d}: {len(train_df):,}; columns: {len(train_df.columns):,}")
    clf, reg, bucket_model, bucket_values, feature_columns, cat_features = tp.train_models(train_df, args)
    args.end_month = original_end_month
    return {
        "clf": clf,
        "reg": reg,
        "bucket_model": bucket_model,
        "bucket_values": bucket_values,
        "feature_columns": feature_columns,
        "cat_features": cat_features,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="")
    parser.add_argument("--runs-dir", default=str(OUTPUT_DIR / "runs"))
    parser.add_argument("--bad-day-mode", choices=["threshold", "known", "none"], default="threshold")
    parser.add_argument("--start-month", type=int, default=4)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument("--validation-month", type=int, default=12)
    parser.add_argument("--validation-months", type=int, nargs="+", default=[12])
    parser.add_argument("--max-positive-per-month", type=int, default=120_000)
    parser.add_argument("--max-negative-per-month", type=int, default=240_000)
    parser.add_argument("--negative-ratio", type=int, default=2)
    parser.add_argument("--validation-positive-rows", type=int, default=300_000)
    parser.add_argument("--classifier-iterations", type=int, default=700)
    parser.add_argument("--regressor-iterations", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--regressor-loss", default="MAPE")
    parser.add_argument("--regressor-target", choices=["log", "raw"], default="raw")
    parser.add_argument("--use-bucket-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bucket-iterations", type=int, default=500)
    parser.add_argument("--mape-weight-power", type=float, default=0.0)
    parser.add_argument("--mape-weight-floor", type=float, default=1.0)
    parser.add_argument("--classifier-positive-weight", type=float, default=1.0)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--location-batch-size", type=int, default=40)
    parser.add_argument("--max-prediction-locations", type=int, default=0)
    parser.add_argument("--threshold-grid", type=float, nargs="+", default=[0.0])
    parser.add_argument("--scale-grid", type=float, nargs="+", default=[0.5, 0.65, 0.75, 0.85, 1.0])
    parser.add_argument("--blend-grid", type=float, nargs="+", default=[0.0])
    parser.add_argument("--bucket-blend-grid", type=float, nargs="+", default=[0.0])
    parser.add_argument("--proba-power-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--final-threshold", type=float, default=0.0)
    parser.add_argument("--final-scale", type=float, default=0.75)
    parser.add_argument("--final-blend", type=float, default=0.0)
    parser.add_argument("--final-bucket-blend", type=float, default=0.0)
    parser.add_argument("--final-proba-power", type=float, default=0.0)
    parser.add_argument(
        "--prediction-strategy",
        default="catboost",
        choices=[
            "catboost",
            "pair_last_30d",
            "pair_last_90d_daily",
            "pair_last_180d_daily",
            "pair_blend_recent",
            "item_share_90d",
            "item_share_smooth",
            "conservative_min",
        ],
    )
    parser.add_argument("--baseline-scale-grid", type=float, nargs="+", default=[0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15])
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    tp.validate_inputs(args)

    run_root = Path(args.runs_dir) / (args.run_name or run_id())
    december_dir = run_root / "december_2025"
    january_dir = run_root / "january_2026"
    models_dir = run_root / "models"
    for path in [december_dir, january_dir, models_dir]:
        path.mkdir(parents=True, exist_ok=True)
    tp.MODEL_DIR = models_dir

    bad_days = set(tp.detect_bad_days(args.bad_day_mode))
    locations = tp.candidate_locations()
    active_items = tp.active_items_with_metadata()
    log(f"Run folder: {run_root}")
    log(f"Bad days: {[d.isoformat() for d in sorted(bad_days)]}")
    log(f"Locations: {locations.height:,}; active items: {active_items.height:,}; candidates: {locations.height * active_items.height:,}")

    # Train only on months before December, tune on December positives, and write a full-grid December submission.
    validation_bundle = train_bundle(args, bad_days, locations, active_items, end_month=11)
    tune = tp.tune_on_validation(
        args,
        bad_days,
        active_items,
        validation_bundle["clf"],
        validation_bundle["reg"],
        validation_bundle["bucket_model"],
        validation_bundle["bucket_values"],
        validation_bundle["feature_columns"],
        validation_bundle["cat_features"],
    )
    validation_bundle["tune"] = tune
    dec_submission = predict_candidate_grid(
        args,
        bad_days,
        locations,
        active_items,
        validation_bundle,
        cutoff=date(2025, 11, 30),
        target_month=12,
        target_days=31,
        output_path=december_dir / "submission.pkl",
    )
    metrics = score_december_submission(dec_submission, bad_days, active_items)

    # Retrain through December and predict January 2026 using the December-tuned parameters.
    final_bundle = train_bundle(args, bad_days, locations, active_items, end_month=12)
    final_bundle["tune"] = tune
    jan_submission = predict_candidate_grid(
        args,
        bad_days,
        locations,
        active_items,
        final_bundle,
        cutoff=date(2025, 12, 31),
        target_month=1,
        target_days=31,
        output_path=january_dir / "submission.pkl",
    )

    validation_bundle["clf"].save_model(models_dir / "validation_positive_classifier.cbm")
    validation_bundle["reg"].save_model(models_dir / "validation_quantity_regressor.cbm")
    final_bundle["clf"].save_model(models_dir / "final_positive_classifier.cbm")
    final_bundle["reg"].save_model(models_dir / "final_quantity_regressor.cbm")

    metrics.update(
        {
            "run_root": str(run_root),
            "december_submission": str(december_dir / "submission.pkl"),
            "january_submission": str(january_dir / "submission.pkl"),
            "january_rows": int(len(jan_submission)),
            "january_nonzero_rows": int((jan_submission["quantity"] > 0).sum()),
            "january_pred_sum": int(jan_submission["quantity"].sum()),
            "tuning": tune,
            "args": vars(args),
        }
    )
    (run_root / "metrics.txt").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()