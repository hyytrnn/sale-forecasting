"""Build the daily cache and reproducible EDA artifacts for the sales data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    return lazy_frame.collect(engine="streaming")


def build_daily_cache(data_dir: Path, cache_path: Path) -> None:
    transactions = data_dir / "transaction_full_2025.parquet"
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    (
        pl.scan_parquet(transactions)
        .filter(pl.col("event_type") == "Purchase")
        .select(
            pl.col("location").cast(pl.Int32),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("updated_date").dt.date().alias("date"),
            pl.col("quantity").cast(pl.Float64),
        )
        .group_by(["location", "item_id", "date"])
        .agg(pl.col("quantity").sum().alias("quantity"))
        .sink_parquet(cache_path)
    )


def save_bar(frame: pl.DataFrame, x: str, y: str, title: str, path: Path, rotation: int = 45) -> None:
    plt.figure(figsize=(12, 6))
    plt.bar(frame[x].to_list(), frame[y].to_list())
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_line(frame: pl.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(frame[x].to_list(), frame[y].to_list(), linewidth=1.2)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.data_dir / "daily_quantity.parquet"
    build_daily_cache(args.data_dir, cache_path)

    daily = pl.scan_parquet(cache_path).filter(pl.col("quantity") > 0)
    overview = collect_streaming(
        daily.select(
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
            pl.len().alias("nonzero_pair_days"),
            pl.struct(["location", "item_id"]).n_unique().alias("pairs"),
            pl.col("location").n_unique().alias("locations"),
            pl.col("item_id").n_unique().alias("items"),
            pl.col("quantity").sum().alias("total_quantity"),
            pl.col("quantity").mean().alias("mean_quantity"),
            pl.col("quantity").median().alias("median_quantity"),
            pl.col("quantity").quantile(0.90).alias("p90_quantity"),
            pl.col("quantity").quantile(0.99).alias("p99_quantity"),
            pl.col("quantity").max().alias("max_quantity"),
        )
    )
    overview.write_json(args.output_dir / "overview.json")
    overview.write_csv(args.output_dir / "overview.csv")

    pair_summary = collect_streaming(
        daily.group_by(["location", "item_id"])
        .agg(
            pl.col("date").min().alias("first_sold_date"),
            pl.col("date").max().alias("last_sold_date"),
            pl.col("date").n_unique().alias("active_days"),
            pl.col("quantity").sum().alias("total_quantity"),
            pl.col("quantity").mean().alias("mean_quantity"),
            pl.col("quantity").median().alias("median_quantity"),
            pl.col("quantity").std().fill_null(0).alias("std_quantity"),
            pl.col("quantity").quantile(0.25).alias("q25_quantity"),
            pl.col("quantity").quantile(0.75).alias("q75_quantity"),
            pl.col("quantity").max().alias("max_quantity"),
        )
        .with_columns(
            (
                pl.col("std_quantity") / pl.col("mean_quantity").clip(lower_bound=1e-9)
            ).alias("coefficient_of_variation"),
            (
                pl.col("q75_quantity")
                + 1.5 * (pl.col("q75_quantity") - pl.col("q25_quantity"))
            ).alias("high_outlier_threshold"),
        )
    )
    pair_summary.sort("total_quantity", descending=True).head(args.top_n).write_csv(
        args.output_dir / "top_pairs_by_total_quantity.csv"
    )
    pair_summary.sort("active_days", descending=True).head(args.top_n).write_csv(
        args.output_dir / "top_pairs_by_active_days.csv"
    )
    pair_summary.filter(pl.col("active_days") >= 10).sort(
        "coefficient_of_variation", descending=True
    ).head(args.top_n).write_csv(args.output_dir / "volatile_pairs.csv")

    daily_with_calendar = daily.with_columns(
        pl.col("date").dt.weekday().alias("weekday"),
        pl.col("date").dt.month().alias("month"),
    )
    date_summary = collect_streaming(
        daily_with_calendar.group_by("date")
        .agg(
            pl.col("quantity").sum().alias("total_quantity"),
            pl.len().alias("nonzero_pair_days"),
            pl.col("location").n_unique().alias("active_locations"),
            pl.col("item_id").n_unique().alias("active_items"),
        )
        .sort("date")
    )
    date_summary.write_csv(args.output_dir / "date_summary.csv")

    weekday_summary = collect_streaming(
        daily_with_calendar.group_by("weekday")
        .agg(
            pl.col("quantity").sum().alias("total_quantity"),
            pl.col("quantity").mean().alias("mean_quantity"),
            pl.len().alias("nonzero_pair_days"),
        )
        .sort("weekday")
    )
    weekday_summary.write_csv(args.output_dir / "weekday_summary.csv")

    location_summary = collect_streaming(
        daily.group_by("location")
        .agg(
            pl.col("quantity").sum().alias("total_quantity"),
            pl.len().alias("nonzero_pair_days"),
            pl.col("item_id").n_unique().alias("items_sold"),
        )
        .sort("total_quantity", descending=True)
    )
    location_summary.head(args.top_n).write_csv(args.output_dir / "top_locations.csv")

    item_summary = collect_streaming(
        daily.group_by("item_id")
        .agg(
            pl.col("quantity").sum().alias("total_quantity"),
            pl.len().alias("nonzero_pair_days"),
            pl.col("location").n_unique().alias("locations_sold"),
        )
        .sort("total_quantity", descending=True)
    )
    item_summary.head(args.top_n).write_csv(args.output_dir / "top_items.csv")

    quantities = collect_streaming(daily.select("quantity").limit(2_000_000))["quantity"].to_list()
    plt.figure(figsize=(12, 6))
    plt.hist(quantities, bins=80, log=True)
    plt.title("Sold-day quantity distribution")
    plt.xlabel("Quantity")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "quantity_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(pair_summary["active_days"].to_list(), bins=80, log=True)
    plt.title("Active sold-day count per location-item pair")
    plt.xlabel("Active days")
    plt.ylabel("Pairs (log scale)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "pair_active_days_distribution.png", dpi=160)
    plt.close()

    save_line(
        date_summary,
        "date",
        "total_quantity",
        "Total sold quantity by day",
        args.output_dir / "daily_total_quantity.png",
    )
    save_bar(
        weekday_summary,
        "weekday",
        "total_quantity",
        "Total quantity by weekday",
        args.output_dir / "weekday_quantity.png",
        rotation=0,
    )
    save_bar(
        location_summary.head(args.top_n),
        "location",
        "total_quantity",
        f"Top {args.top_n} locations by quantity",
        args.output_dir / "top_locations_quantity.png",
    )
    save_bar(
        item_summary.head(args.top_n),
        "item_id",
        "total_quantity",
        f"Top {args.top_n} items by quantity",
        args.output_dir / "top_items_quantity.png",
    )

    print(json.dumps(overview.to_dicts()[0], default=str, indent=2))
    print(f"EDA outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
