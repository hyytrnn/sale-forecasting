import argparse
from pathlib import Path

import polars as pl

from .project_paths import CACHE_DIR, RAW_DATA_DIR


def log(message: str) -> None:
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="subset-data")
    parser.add_argument("--locations", type=int, default=40)
    parser.add_argument("--items", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-event", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_out = output_dir / "raw-data"
    cache_out = output_dir / "outputs" / "cache"
    raw_out.mkdir(parents=True, exist_ok=True)
    cache_out.mkdir(parents=True, exist_ok=True)

    items_path = RAW_DATA_DIR / "items.parquet"
    tx_path = RAW_DATA_DIR / "transaction_full_2025.parquet"
    daily_path = CACHE_DIR / "daily_quantity.parquet"

    active_items = (
        pl.scan_parquet(items_path)
        .filter(pl.col("sale_status") == 1)
        .select(pl.col("item_id").cast(pl.Utf8))
        .unique()
        .collect()
        .sample(n=args.items, seed=args.seed, shuffle=True)
    )
    locations = (
        pl.scan_parquet(tx_path)
        .select("location")
        .unique()
        .collect()
        .sample(n=args.locations, seed=args.seed + 1, shuffle=True)
    )

    log(f"Subset locations={locations.height:,}, active_items={active_items.height:,}")
    log(f"Candidate pairs={locations.height * active_items.height:,}")

    items_subset = (
        pl.scan_parquet(items_path)
        .with_columns(pl.col("item_id").cast(pl.Utf8))
        .join(active_items.lazy(), on="item_id", how="inner")
        .collect()
    )
    tx_subset = (
        pl.scan_parquet(tx_path)
        .with_columns(pl.col("item_id").cast(pl.Utf8))
        .join(active_items.lazy(), on="item_id", how="inner")
        .join(locations.lazy(), on="location", how="inner")
        .collect()
    )
    daily_subset = (
        pl.scan_parquet(daily_path)
        .with_columns(pl.col("item_id").cast(pl.Utf8))
        .join(active_items.lazy(), on="item_id", how="inner")
        .join(locations.lazy(), on="location", how="inner")
        .collect()
    )

    items_subset.write_parquet(raw_out / "items.parquet")
    tx_subset.write_parquet(raw_out / "transaction_full_2025.parquet")
    daily_subset.write_parquet(cache_out / "daily_quantity.parquet")

    if args.include_event:
        event_path = RAW_DATA_DIR / "event_full_2025.parquet"
        event_subset = (
            pl.scan_parquet(event_path)
            .with_columns(pl.col("item_id").cast(pl.Utf8))
            .join(active_items.lazy(), on="item_id", how="inner")
            .collect()
        )
        event_subset.write_parquet(raw_out / "event_full_2025.parquet")

    log(f"Wrote {raw_out / 'items.parquet'} rows={items_subset.height:,}")
    log(f"Wrote {raw_out / 'transaction_full_2025.parquet'} rows={tx_subset.height:,}")
    log(f"Wrote {cache_out / 'daily_quantity.parquet'} rows={daily_subset.height:,}")
    log("")
    log("Run local subset test:")
    log(f"  SALES_FORECASTING_DATA_DIR={raw_out.resolve()}")
    log(f"  SALES_FORECASTING_OUTPUT_DIR={(output_dir / 'outputs').resolve()}")
    log("  python -m sales_forecasting.train_predict_decval_v2 --start-month 10 --end-month 11 --validation-months 11 --max-positive-per-month 2000 --max-negative-per-month 4000 --validation-positive-rows 2000 --classifier-iterations 20 --regressor-iterations 20 --location-batch-size 10")


if __name__ == "__main__":
    main()
