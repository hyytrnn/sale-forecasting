import argparse
import calendar
import json
import pickle
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from .project_paths import CACHE_DIR, OUTPUT_DIR, RAW_DATA_DIR, ensure_output_dirs


TRANSACTIONS_PATH = RAW_DATA_DIR / "transaction_full_2025.parquet"
ITEMS_PATH = RAW_DATA_DIR / "items.parquet"
DAILY_PATH = CACHE_DIR / "daily_quantity.parquet"
SUBMISSION_PATH = OUTPUT_DIR / "submission.pkl"
MODEL_DIR = OUTPUT_DIR / "models"

BAD_DAY_QTY_THRESHOLD = 100_000
KNOWN_BAD_DAYS = [
    date(2025, 1, 29),
    date(2025, 1, 30),
    date(2025, 1, 31),
    date(2025, 7, 4),
    date(2025, 7, 18),
    date(2025, 9, 29),
    date(2025, 10, 27),
]
RECENT_WINDOWS = (30, 60, 90, 180, 365)
CAT_COLUMNS = [
    "location",
    "item_id",
    "category_l1",
    "category_l2",
    "category_l3",
    "category",
    "brand",
    "manufacturer",
    "size",
]
ITEM_META_COLUMNS = ["category_l1", "category_l2", "category_l3", "category", "brand", "manufacturer"]


def log(message: str) -> None:
    print(message, flush=True)


def step_timer(name: str):
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            log(f"START {name}")

        def __exit__(self, exc_type, exc, traceback):
            elapsed = time.perf_counter() - self.start
            status = "FAILED" if exc_type else "DONE"
            log(f"{status} {name} in {elapsed:.1f}s")

    return Timer()


def month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def ensure_daily_cache() -> None:
    if DAILY_PATH.exists():
        return
    log(f"Missing {DAILY_PATH}; building daily cache from raw transactions")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (
        pl.scan_parquet(TRANSACTIONS_PATH)
        .filter(pl.col("event_type") == "Purchase")
        .select(
            "location",
            pl.col("item_id").cast(pl.Utf8).alias("item_id"),
            pl.col("updated_date").dt.date().alias("date"),
            pl.col("quantity").cast(pl.Float32).alias("quantity"),
        )
        .group_by(["location", "item_id", "date"])
        .agg(pl.col("quantity").sum().alias("y"))
        .sink_parquet(DAILY_PATH)
    )


def validate_inputs(args) -> None:
    missing = [path for path in [TRANSACTIONS_PATH, ITEMS_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    ensure_daily_cache()
    if args.task_type == "GPU":
        log("Running CatBoost GPU preflight")
        try:
            toy_x = pd.DataFrame({"x": [0, 1, 2, 3], "cat": ["a", "b", "a", "b"]})
            toy_y = np.array([0, 1, 0, 1])
            toy_model = CatBoostClassifier(
                iterations=2,
                depth=2,
                learning_rate=0.1,
                loss_function="Logloss",
                task_type="GPU",
                devices=args.devices,
                verbose=False,
            )
            toy_model.fit(Pool(toy_x, toy_y, cat_features=["cat"]))
        except Exception as exc:
            raise RuntimeError(
                "CatBoost GPU preflight failed before expensive training. "
                "Switch to --task-type CPU or fix the Colab CUDA/CatBoost environment."
            ) from exc


def detect_bad_days(mode: str) -> list[date]:
    if mode == "none":
        return []
    if mode == "known":
        return KNOWN_BAD_DAYS
    daily_totals = (
        pl.scan_parquet(DAILY_PATH)
        .group_by("date")
        .agg(pl.col("y").sum().alias("qty"))
        .collect()
    )
    calendar_df = pl.DataFrame(
        {"date": pl.date_range(date(2025, 1, 1), date(2025, 12, 31), interval="1d", eager=True)}
    )
    full = calendar_df.join(daily_totals, on="date", how="left").with_columns(pl.col("qty").fill_null(0))
    bad_days = sorted(full.filter(pl.col("qty") < BAD_DAY_QTY_THRESHOLD)["date"].to_list())
    if len(bad_days) > 30:
        log(
            f"Threshold anomaly detection found {len(bad_days)} bad days, "
            "likely because this is a subset. Falling back to known full-data anomaly dates."
        )
        return KNOWN_BAD_DAYS
    return bad_days


def valid_days_between(start: date, end: date, bad_days: set[date]) -> int:
    days = 0
    current = start
    while current <= end:
        if current not in bad_days:
            days += 1
        current += timedelta(days=1)
    return days


def candidate_locations() -> pl.DataFrame:
    return (
        pl.scan_parquet(TRANSACTIONS_PATH)
        .select(pl.col("location").cast(pl.Utf8).alias("location"))
        .unique()
        .sort("location")
        .collect()
    )


def active_items_with_metadata() -> pl.DataFrame:
    text_defaults = {
        "category_l1": "unknown",
        "category_l2": "unknown",
        "category_l3": "unknown",
        "category": "unknown",
        "brand": "unknown",
        "manufacturer": "unknown",
        "size": "unknown",
    }
    return (
        pl.scan_parquet(ITEMS_PATH)
        .filter(pl.col("sale_status") == 1)
        .select(
            pl.col("item_id").cast(pl.Utf8),
            pl.col("price").cast(pl.Float64).fill_null(0).alias("price"),
            *[
                pl.col(col).cast(pl.Utf8).fill_null(default).alias(col)
                for col, default in text_defaults.items()
            ],
        )
        .unique("item_id")
        .sort("item_id")
        .collect()
    )


def daily_valid_lf(bad_days: set[date]) -> pl.LazyFrame:
    return (
        pl.scan_parquet(DAILY_PATH)
        .select(
            pl.col("location").cast(pl.Utf8).alias("location"),
            pl.col("item_id").cast(pl.Utf8).alias("item_id"),
            pl.col("date").cast(pl.Date),
            pl.col("y").cast(pl.Float64).alias("quantity"),
        )
        .filter(~pl.col("date").is_in(list(bad_days)))
    )


def monthly_target(
    daily_lf: pl.LazyFrame,
    active_items: pl.DataFrame,
    target_year: int,
    target_month: int,
    bad_days: set[date],
) -> pl.DataFrame:
    start, end = month_bounds(target_year, target_month)
    valid_days = valid_days_between(start, end, bad_days)
    calendar_days = (end - start).days + 1
    return (
        daily_lf.filter((pl.col("date") >= start) & (pl.col("date") <= end))
        .group_by(["location", "item_id"])
        .agg(pl.col("quantity").sum().alias("quantity"))
        .join(active_items.lazy().select("item_id"), on="item_id", how="inner")
        .with_columns((pl.col("quantity") * calendar_days / valid_days).alias("quantity"))
        .collect()
    )


def sample_frame(df: pl.DataFrame, max_rows: int, seed: int) -> pl.DataFrame:
    if df.height <= max_rows:
        return df
    return df.sample(n=max_rows, seed=seed, shuffle=True)


def negative_sample(
    locations: pl.DataFrame,
    active_items: pl.DataFrame,
    all_positives: pl.DataFrame,
    n_rows: int,
    seed: int,
    max_attempts: int = 100,
) -> pl.DataFrame:
    """Return unique zero-demand pairs that never overlap observed positives."""
    key_columns = ["location", "item_id"]
    if n_rows <= 0:
        return pl.DataFrame(
            schema={"location": pl.Utf8, "item_id": pl.Utf8, "quantity": pl.Float64}
        )
    rng = np.random.default_rng(seed)
    loc_pd = locations.to_pandas()
    item_pd = active_items.select("item_id").to_pandas()
    positive_keys = all_positives.select(key_columns).unique()
    candidate_count = locations["location"].n_unique() * active_items["item_id"].n_unique()
    target_rows = min(n_rows, max(candidate_count - positive_keys.height, 0))
    selected = pl.DataFrame(schema={"location": pl.Utf8, "item_id": pl.Utf8})
    attempts = 0
    while selected.height < target_rows and attempts < max_attempts:
        remaining = target_rows - selected.height
        draw = max(remaining * 3, 50_000)
        sampled_locations = loc_pd.iloc[rng.integers(0, len(loc_pd), size=draw)].reset_index(drop=True)
        sampled_items = item_pd.iloc[rng.integers(0, len(item_pd), size=draw)].reset_index(drop=True)
        candidates = pl.from_pandas(
            pd.DataFrame(
                {
                    "location": sampled_locations["location"].astype(str),
                    "item_id": sampled_items["item_id"].astype(str),
                }
            )
        ).unique()
        negatives = candidates.join(positive_keys, on=key_columns, how="anti")
        if selected.height:
            negatives = negatives.join(selected, on=key_columns, how="anti")
        if negatives.height:
            selected = pl.concat(
                [selected, negatives.head(remaining)], how="vertical"
            ).unique()
        attempts += 1
    if selected.height < n_rows:
        log(
            f"Requested {n_rows:,} unique negatives but candidate sampling found "
            f"{selected.height:,} after {attempts} attempts"
        )
    result = selected.head(n_rows).with_columns(pl.lit(0.0).alias("quantity"))
    overlap = result.select(key_columns).join(positive_keys, on=key_columns, how="inner")
    if overlap.height:
        raise AssertionError("Negative sampling produced positive location-item keys")
    return result


def aggregate_features(daily_lf: pl.LazyFrame, cutoff: date, prefix: str, keys: list[str]) -> pl.DataFrame:
    lf = daily_lf.filter(pl.col("date") <= cutoff)
    aggs = [
        pl.col("quantity").sum().alias(f"{prefix}_qty_all"),
        pl.col("date").n_unique().alias(f"{prefix}_active_days_all"),
        pl.col("date").max().alias(f"{prefix}_last_sale_date"),
    ]
    for window in RECENT_WINDOWS:
        start = cutoff - timedelta(days=window - 1)
        in_window = pl.col("date") >= start
        aggs.extend(
            [
                pl.col("quantity").filter(in_window).sum().alias(f"{prefix}_qty_{window}d"),
                pl.col("date").filter(in_window).n_unique().alias(f"{prefix}_active_days_{window}d"),
            ]
        )
    return lf.group_by(keys).agg(aggs).collect()


def aggregate_meta_features(
    daily_lf: pl.LazyFrame,
    active_items: pl.DataFrame,
    cutoff: date,
    prefix: str,
    keys: list[str],
) -> pl.DataFrame:
    meta_lf = active_items.lazy().select(["item_id", *ITEM_META_COLUMNS])
    return aggregate_features(daily_lf.join(meta_lf, on="item_id", how="left"), cutoff, prefix, keys)


def build_feature_tables(daily_lf: pl.LazyFrame, active_items: pl.DataFrame, cutoff: date) -> dict[str, pl.DataFrame]:
    with step_timer(f"feature tables cutoff={cutoff}"):
        return {
            "pair": aggregate_features(daily_lf, cutoff, "pair", ["location", "item_id"]),
            "item": aggregate_features(daily_lf, cutoff, "item", ["item_id"]),
            "location": aggregate_features(daily_lf, cutoff, "location", ["location"]),
            "category_l1": aggregate_meta_features(daily_lf, active_items, cutoff, "cat1", ["category_l1"]),
            "category_l2": aggregate_meta_features(daily_lf, active_items, cutoff, "cat2", ["category_l2"]),
            "brand": aggregate_meta_features(daily_lf, active_items, cutoff, "brand", ["brand"]),
            "location_category_l1": aggregate_meta_features(
                daily_lf, active_items, cutoff, "loc_cat1", ["location", "category_l1"]
            ),
        }


def add_features_from_tables(
    pairs: pl.DataFrame,
    active_items: pl.DataFrame,
    feature_tables: dict[str, pl.DataFrame],
    cutoff: date,
    target_month: int,
    target_days: int,
) -> pd.DataFrame:
    features = (
        pairs.select(["location", "item_id", "quantity"] if "quantity" in pairs.columns else ["location", "item_id"])
        .join(active_items, on="item_id", how="left")
        .join(feature_tables["pair"], on=["location", "item_id"], how="left")
        .join(feature_tables["item"], on="item_id", how="left")
        .join(feature_tables["location"], on="location", how="left")
        .join(feature_tables["category_l1"], on="category_l1", how="left")
        .join(feature_tables["category_l2"], on="category_l2", how="left")
        .join(feature_tables["brand"], on="brand", how="left")
        .join(feature_tables["location_category_l1"], on=["location", "category_l1"], how="left")
        .with_columns(
            pl.lit(target_month).alias("target_month"),
            pl.lit(target_days).alias("target_days"),
        )
    )

    date_columns = [col for col in features.columns if col.endswith("_last_sale_date")]
    for col in date_columns:
        features = features.with_columns(
            (pl.lit(cutoff) - pl.col(col)).dt.total_days().fill_null(999).alias(col.replace("_last_sale_date", "_days_since_sale"))
        ).drop(col)

    numeric_columns = [col for col, dtype in zip(features.columns, features.dtypes) if dtype.is_numeric()]
    features = features.with_columns(pl.col(numeric_columns).fill_null(0))
    features = features.with_columns(
        (pl.col("pair_qty_30d") / (pl.col("pair_qty_90d") + 1)).alias("pair_recent_share_30_90"),
        (pl.col("item_qty_30d") / (pl.col("item_qty_90d") + 1)).alias("item_recent_share_30_90"),
        (pl.col("location_qty_30d") / (pl.col("location_qty_90d") + 1)).alias("location_recent_share_30_90"),
        (pl.col("pair_qty_90d") / (pl.col("item_qty_90d") + 1)).alias("pair_item_share_90d"),
        (pl.col("pair_qty_90d") / (pl.col("location_qty_90d") + 1)).alias("pair_location_share_90d"),
    )
    for col in CAT_COLUMNS:
        if col in features.columns:
            features = features.with_columns(pl.col(col).cast(pl.Utf8).fill_null("unknown"))
    return features.to_pandas()


def sample_training_pairs(
    target: pl.DataFrame,
    locations: pl.DataFrame,
    active_items: pl.DataFrame,
    args,
    month: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sample labeled pairs without treating unsampled positives as negatives."""
    all_positives = target.filter(pl.col("quantity") > 0)
    positives = sample_frame(
        all_positives,
        args.max_positive_per_month,
        seed=10_000 + month,
    )
    negative_count = min(
        args.max_negative_per_month,
        max(positives.height * args.negative_ratio, 1),
    )
    negatives = negative_sample(
        locations,
        active_items,
        all_positives,
        negative_count,
        seed=20_000 + month,
    )
    return positives, negatives


def build_training_data(args, bad_days: set[date], locations: pl.DataFrame, active_items: pl.DataFrame):
    daily_lf = daily_valid_lf(bad_days)
    frames: list[pd.DataFrame] = []
    months = [month for month in range(args.start_month, args.end_month + 1)]
    for month in months:
        cutoff = month_bounds(2025, month)[0] - timedelta(days=1)
        _, target_end = month_bounds(2025, month)
        target_days = calendar.monthrange(2025, month)[1]
        target = monthly_target(daily_lf, active_items, 2025, month, bad_days)
        positives, negatives = sample_training_pairs(
            target,
            locations,
            active_items,
            args,
            month,
        )
        month_pairs = pl.concat([positives, negatives], how="vertical")
        log(
            f"Month {month:02d}: cutoff={cutoff}, positives={positives.height:,}, "
            f"negatives={negatives.height:,}, target_end={target_end}"
        )
        feature_tables = build_feature_tables(daily_lf, active_items, cutoff)
        month_features = add_features_from_tables(month_pairs, active_items, feature_tables, cutoff, month, target_days)
        frames.append(month_features)
    return pd.concat(frames, ignore_index=True)


def score_metric(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return 0.0
    ape = np.abs(actual[mask] - pred[mask]) / np.abs(actual[mask])
    return round(100.0 - float(ape.mean()) * 100.0, 4)


def mape_sample_weight(y: np.ndarray, args) -> np.ndarray:
    if args.mape_weight_power <= 0:
        return np.ones_like(y, dtype=np.float64)
    clipped = np.clip(y, args.mape_weight_floor, None)
    weights = 1.0 / np.power(clipped, args.mape_weight_power)
    return weights / weights.mean()


def quantity_bucket_labels(y: np.ndarray) -> np.ndarray:
    rounded = np.rint(np.clip(y, 1, None)).astype(np.int64)
    labels = np.zeros_like(rounded)
    labels[rounded == 1] = 0
    labels[rounded == 2] = 1
    labels[rounded == 3] = 2
    labels[(rounded >= 4) & (rounded <= 5)] = 3
    labels[(rounded >= 6) & (rounded <= 10)] = 4
    labels[(rounded >= 11) & (rounded <= 20)] = 5
    labels[(rounded >= 21) & (rounded <= 50)] = 6
    labels[rounded >= 51] = 7
    return labels


def bucket_values_from_training(y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    fallback = np.array([1, 2, 3, 4.5, 8, 15, 30, 60], dtype=np.float64)
    values = fallback.copy()
    for label in range(len(values)):
        bucket_y = y[labels == label]
        if len(bucket_y):
            values[label] = float(np.median(bucket_y))
    return values


def train_models(train_df: pd.DataFrame, args):
    y = train_df.pop("quantity").astype(float).to_numpy()
    positive = (y > 0).astype(int)
    cat_features = [col for col in CAT_COLUMNS if col in train_df.columns]

    clf = CatBoostClassifier(
        iterations=args.classifier_iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.seed,
        thread_count=args.thread_count,
        task_type=args.task_type,
        devices=args.devices,
        verbose=100,
    )
    with step_timer("train positive classifier"):
        class_weights = np.where(positive == 1, args.classifier_positive_weight, 1.0)
        clf.fit(Pool(train_df, positive, cat_features=cat_features, weight=class_weights))

    reg = CatBoostRegressor(
        iterations=args.regressor_iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        loss_function=args.regressor_loss,
        random_seed=args.seed,
        thread_count=args.thread_count,
        task_type=args.task_type,
        devices=args.devices,
        verbose=100,
    )
    positive_df = train_df.loc[positive == 1].copy()
    positive_y = y[positive == 1]
    if args.regressor_target == "log":
        reg_target = np.log1p(positive_y)
    else:
        reg_target = positive_y
    reg_weights = mape_sample_weight(positive_y, args)
    with step_timer("train positive quantity regressor"):
        reg.fit(Pool(positive_df, reg_target, cat_features=cat_features, weight=reg_weights))

    bucket_model = None
    bucket_values = None
    if args.use_bucket_model:
        bucket_labels = quantity_bucket_labels(positive_y)
        bucket_values = bucket_values_from_training(positive_y, bucket_labels)
        if len(np.unique(bucket_labels)) >= 2:
            bucket_model = CatBoostClassifier(
                iterations=args.bucket_iterations,
                learning_rate=args.learning_rate,
                depth=args.depth,
                loss_function="MultiClass",
                random_seed=args.seed,
                thread_count=args.thread_count,
                task_type=args.task_type,
                devices=args.devices,
                verbose=100,
            )
            with step_timer("train positive quantity bucket classifier"):
                bucket_model.fit(Pool(positive_df, bucket_labels, cat_features=cat_features, weight=reg_weights))
        else:
            log("Skipping bucket classifier: only one positive quantity bucket in training data")

    return clf, reg, bucket_model, bucket_values, list(train_df.columns), cat_features


def predict_raw_components(
    args,
    clf,
    reg,
    bucket_model,
    bucket_values,
    features: pd.DataFrame,
    feature_columns: list[str],
    cat_features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x = features[feature_columns]
    pool = Pool(x, cat_features=cat_features)
    proba = clf.predict_proba(pool)[:, 1]
    reg_pred = reg.predict(pool)
    amount = np.expm1(reg_pred) if args.regressor_target == "log" else reg_pred
    amount = np.clip(amount, 0, None)
    bucket_amount = None
    if bucket_model is not None and bucket_values is not None:
        bucket_proba = bucket_model.predict_proba(pool)
        class_count = min(bucket_proba.shape[1], len(bucket_values))
        bucket_amount = bucket_proba[:, :class_count] @ bucket_values[:class_count]
    return proba, amount, bucket_amount


def predict_quantity(
    args,
    clf,
    reg,
    bucket_model,
    bucket_values,
    features: pd.DataFrame,
    feature_columns: list[str],
    cat_features: list[str],
    threshold: float,
    scale: float,
    blend: float,
    bucket_blend: float,
    proba_power: float,
) -> np.ndarray:
    proba, amount, bucket_amount = predict_raw_components(
        args, clf, reg, bucket_model, bucket_values, features, feature_columns, cat_features
    )
    if bucket_amount is not None and bucket_blend > 0:
        amount = (1.0 - bucket_blend) * amount + bucket_blend * bucket_amount
    baseline = baseline_predictions(features)["pair_last_90d_daily"].astype(float)
    proba_factor = np.power(np.clip(proba, 0, 1), proba_power)
    pred = proba_factor * amount * scale
    if blend > 0:
        pred = (1.0 - blend) * pred + blend * baseline
    if threshold > 0:
        pred[proba < threshold] = 0
    pred = np.rint(np.clip(pred, 0, None)).astype(np.int64)
    return pred


def apply_baseline_strategy(features: pd.DataFrame, strategy: str, scale: float) -> np.ndarray:
    baselines = baseline_predictions(features)
    if strategy not in baselines:
        raise ValueError(f"Unknown baseline strategy {strategy}. Options: {sorted(baselines)}")
    pred = baselines[strategy].astype(float) * scale
    return np.rint(np.clip(pred, 0, None)).astype(np.int64)


def baseline_predictions(features: pd.DataFrame) -> dict[str, np.ndarray]:
    target_days = features["target_days"].to_numpy()
    pair_last_30 = np.rint(np.clip(features["pair_qty_30d"].to_numpy(), 0, None)).astype(np.int64)
    pair_last_90_daily = features["pair_qty_90d"].to_numpy() / 90.0
    pair_last_90 = np.rint(np.clip(pair_last_90_daily * target_days, 0, None)).astype(np.int64)
    pair_last_180_daily = features["pair_qty_180d"].to_numpy() / 180.0
    pair_last_180 = np.rint(np.clip(pair_last_180_daily * target_days, 0, None)).astype(np.int64)
    pair_blend = (
        0.50 * features["pair_qty_30d"].to_numpy()
        + 0.30 * pair_last_90_daily * target_days
        + 0.20 * pair_last_180_daily * target_days
    )
    item_share = features["pair_item_share_90d"].to_numpy()
    item_scaled = features["item_qty_30d"].to_numpy() * item_share
    item_share_pred = np.rint(np.clip(item_scaled, 0, None)).astype(np.int64)
    item_share_smooth = (
        (features["pair_qty_90d"].to_numpy() + 0.03 * features["loc_cat1_qty_90d"].to_numpy())
        / (features["item_qty_90d"].to_numpy() + 0.03 * features["cat1_qty_90d"].to_numpy() + 1.0)
    )
    item_smooth_pred = features["item_qty_30d"].to_numpy() * item_share_smooth
    conservative = np.minimum(pair_blend, item_smooth_pred)
    return {
        "pair_last_30d": pair_last_30,
        "pair_last_90d_daily": pair_last_90,
        "pair_last_180d_daily": pair_last_180,
        "pair_blend_recent": np.rint(np.clip(pair_blend, 0, None)).astype(np.int64),
        "item_share_90d": item_share_pred,
        "item_share_smooth": np.rint(np.clip(item_smooth_pred, 0, None)).astype(np.int64),
        "conservative_min": np.rint(np.clip(conservative, 0, None)).astype(np.int64),
    }


def validation_features_for_month(
    args,
    bad_days: set[date],
    active_items: pl.DataFrame,
    month: int,
    positive_only: bool,
) -> pd.DataFrame:
    daily_lf = daily_valid_lf(bad_days)
    cutoff = month_bounds(2025, month)[0] - timedelta(days=1)
    target_days = calendar.monthrange(2025, month)[1]
    target = monthly_target(daily_lf, active_items, 2025, month, bad_days)
    validation_pairs = target.filter(pl.col("quantity") > 0) if positive_only else target
    validation_pairs = sample_frame(validation_pairs, args.validation_positive_rows, seed=30_000 + month)
    feature_tables = build_feature_tables(daily_lf, active_items, cutoff)
    return add_features_from_tables(validation_pairs, active_items, feature_tables, cutoff, month, target_days)


def tune_on_validation(args, bad_days: set[date], active_items: pl.DataFrame, clf, reg, bucket_model, bucket_values, feature_columns, cat_features):
    validation_months = args.validation_months or [args.validation_month]
    validation_frames = [
        validation_features_for_month(args, bad_days, active_items, month, positive_only=True)
        for month in validation_months
    ]
    features = pd.concat(validation_frames, ignore_index=True)
    actual = features["quantity"].astype(float).to_numpy()
    features_no_y = features.drop(columns=["quantity"])

    best = {"score": -1e18, "threshold": 0.0, "scale": 1.0, "blend": 0.0, "bucket_blend": 0.0, "proba_power": 1.0}
    report_rows = []
    for baseline_name in baseline_predictions(features_no_y):
        for baseline_scale in args.baseline_scale_grid:
            pred = apply_baseline_strategy(features_no_y, baseline_name, baseline_scale)
            score = score_metric(actual, pred)
            report_rows.append(
                {
                    "model": baseline_name,
                    "months": ",".join(str(month) for month in validation_months),
                    "threshold": None,
                    "scale": baseline_scale,
                    "blend": None,
                    "bucket_blend": None,
                    "proba_power": None,
                    "score": score,
                    "mean_pred": float(pred.mean()),
                    "nonzero_pred_rate": float((pred > 0).mean()),
                }
            )

    for threshold in args.threshold_grid:
        for scale in args.scale_grid:
            for blend in args.blend_grid:
                for bucket_blend in args.bucket_blend_grid:
                    for proba_power in args.proba_power_grid:
                        pred = predict_quantity(
                            args,
                            clf,
                            reg,
                            bucket_model,
                            bucket_values,
                            features_no_y,
                            feature_columns,
                            cat_features,
                            threshold,
                            scale,
                            blend,
                            bucket_blend,
                            proba_power,
                        )
                        score = score_metric(actual, pred)
                        report_rows.append(
                            {
                                "model": "catboost_two_stage",
                                "months": ",".join(str(month) for month in validation_months),
                                "threshold": threshold,
                                "scale": scale,
                                "blend": blend,
                                "bucket_blend": bucket_blend,
                                "proba_power": proba_power,
                                "score": score,
                                "mean_pred": float(pred.mean()),
                                "nonzero_pred_rate": float((pred > 0).mean()),
                            }
                        )
                        if score > best["score"]:
                            best = {
                                "score": score,
                                "threshold": threshold,
                                "scale": scale,
                                "blend": blend,
                                "bucket_blend": bucket_blend,
                                "proba_power": proba_power,
                            }

    report = pd.DataFrame(report_rows).sort_values("score", ascending=False)
    report_path = MODEL_DIR / "validation_report.csv"
    report.to_csv(report_path, index=False)
    log(f"Validation months {validation_months} best: {best}")
    log(f"Wrote {report_path}")
    log("Top validation rows:")
    log(report.head(8).to_string(index=False))
    return best


def final_prediction(
    args,
    bad_days: set[date],
    locations: pl.DataFrame,
    active_items: pl.DataFrame,
    clf,
    reg,
    bucket_model,
    bucket_values,
    feature_columns,
    cat_features,
    threshold: float,
    scale: float,
    blend: float,
    bucket_blend: float,
    proba_power: float,
):
    daily_lf = daily_valid_lf(bad_days)
    cutoff = date(2025, 12, 31)
    feature_tables = build_feature_tables(daily_lf, active_items, cutoff)
    rows: list[pd.DataFrame] = []
    item_grid = active_items.select("item_id")
    loc_values = locations["location"].to_list()
    if args.max_prediction_locations:
        loc_values = loc_values[: args.max_prediction_locations]
    for start in range(0, len(loc_values), args.location_batch_size):
        batch_locations = pl.DataFrame({"location": loc_values[start : start + args.location_batch_size]})
        pairs = batch_locations.join(item_grid, how="cross")
        features = add_features_from_tables(pairs, active_items, feature_tables, cutoff, 1, 31)
        if args.prediction_strategy == "catboost":
            pred = predict_quantity(
                args,
                clf,
                reg,
                bucket_model,
                bucket_values,
                features,
                feature_columns,
                cat_features,
                threshold,
                scale,
                blend,
                bucket_blend,
                proba_power,
            )
        else:
            pred = apply_baseline_strategy(features, args.prediction_strategy, args.final_scale)
        out = features[["location", "item_id"]].copy()
        out["quantity"] = pred.astype(np.int64)
        rows.append(out)
        log(f"Predicted locations {start + 1:,}-{min(start + args.location_batch_size, len(loc_values)):,} / {len(loc_values):,}")

    submission = pd.concat(rows, ignore_index=True)
    submission = submission[["location", "item_id", "quantity"]]
    submission = pd.DataFrame(
        {
            "location": submission["location"].astype(str).to_numpy(dtype=object),
            "item_id": submission["item_id"].astype(str).to_numpy(dtype=object),
            "quantity": submission["quantity"].astype(np.int64).to_numpy(),
        }
    )
    with SUBMISSION_PATH.open("wb") as f:
        pickle.dump(submission, f)
    return submission


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "final-only", "validate-only"], default="full")
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
    parser.add_argument("--regressor-loss", default="RMSE")
    parser.add_argument("--regressor-target", choices=["log", "raw"], default="log")
    parser.add_argument("--use-bucket-model", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--threshold-grid", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35])
    parser.add_argument("--scale-grid", type=float, nargs="+", default=[0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.25])
    parser.add_argument("--blend-grid", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.35, 0.5])
    parser.add_argument("--bucket-blend-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--proba-power-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--final-threshold", type=float, default=0.02)
    parser.add_argument("--final-scale", type=float, default=1.0)
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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    validate_inputs(args)

    bad_days = set(detect_bad_days(args.bad_day_mode))
    locations = candidate_locations()
    active_items = active_items_with_metadata()
    log(f"Bad days: {[d.isoformat() for d in sorted(bad_days)]}")
    log(f"Locations: {locations.height:,}; active items: {active_items.height:,}; candidates: {locations.height * active_items.height:,}")

    original_end_month = args.end_month
    tune = {
        "score": None,
        "threshold": args.final_threshold,
        "scale": args.final_scale,
        "blend": args.final_blend,
        "bucket_blend": args.final_bucket_blend,
        "proba_power": args.final_proba_power,
    }

    if args.mode in {"full", "validate-only"}:
        validation_months = sorted(args.validation_months or [args.validation_month])
        first_validation_month = validation_months[0]
        if first_validation_month <= args.start_month:
            raise ValueError("first validation month must be greater than --start-month")

        args.end_month = first_validation_month - 1
        train_df = build_training_data(args, bad_days, locations, active_items)
        log(f"Validation-training rows: {len(train_df):,}; columns: {len(train_df.columns):,}")
        clf, reg, bucket_model, bucket_values, feature_columns, cat_features = train_models(train_df, args)
        tune = tune_on_validation(args, bad_days, active_items, clf, reg, bucket_model, bucket_values, feature_columns, cat_features)
        if args.mode == "validate-only":
            return

    args.end_month = original_end_month
    final_train_df = build_training_data(args, bad_days, locations, active_items)
    log(f"Final-training rows: {len(final_train_df):,}; columns: {len(final_train_df.columns):,}")
    clf, reg, bucket_model, bucket_values, feature_columns, cat_features = train_models(final_train_df, args)

    clf.save_model(MODEL_DIR / "catboost_positive_classifier.cbm")
    reg.save_model(MODEL_DIR / "catboost_positive_quantity_regressor.cbm")
    if bucket_model is not None:
        bucket_model.save_model(MODEL_DIR / "catboost_quantity_bucket_classifier.cbm")
    metadata = {
        "bad_days": [d.isoformat() for d in sorted(bad_days)],
        "feature_columns": feature_columns,
        "cat_features": cat_features,
        "bucket_values": bucket_values.tolist() if bucket_values is not None else None,
        "tuning": tune,
        "args": vars(args),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    submission = final_prediction(
        args,
        bad_days,
        locations,
        active_items,
        clf,
        reg,
        bucket_model,
        bucket_values,
        feature_columns,
        cat_features,
        threshold=tune["threshold"],
        scale=tune["scale"],
        blend=tune["blend"],
        bucket_blend=tune["bucket_blend"],
        proba_power=tune["proba_power"],
    )
    log(f"Wrote {SUBMISSION_PATH} with {len(submission):,} rows")
    log(str(submission.dtypes))


if __name__ == "__main__":
    main()
