from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import polars as pl

from .project_paths import CACHE_DIR, OUTPUT_DIR, RAW_DATA_DIR, ensure_output_dirs, first_existing


CACHE_PATH = first_existing(CACHE_DIR / "daily_quantity.parquet", "outputs/daily_quantity.parquet")
TRANSACTIONS_PATH = RAW_DATA_DIR / "transaction_full_2025.parquet"
EDA_DIR = OUTPUT_DIR / "eda"


def log(message):
    print(message, flush=True)


def scan_daily_quantity():
    if Path(CACHE_PATH).exists():
        log(f"Reading daily cache: {CACHE_PATH}")
        return pl.scan_parquet(CACHE_PATH).select(
            pl.col("location"),
            pl.col("item_id").cast(pl.Utf8).alias("item_id"),
            pl.col("date").cast(pl.Date).alias("date"),
            pl.col("y").cast(pl.Float64).alias("quantity"),
        )

    if Path(TRANSACTIONS_PATH).exists():
        log(f"Daily cache not found. Reading raw transactions: {TRANSACTIONS_PATH}")
        return (
            pl.scan_parquet(TRANSACTIONS_PATH)
            .filter(pl.col("event_type") == "Purchase")
            .select(
                "location",
                pl.col("item_id").cast(pl.Utf8).alias("item_id"),
                pl.col("updated_date").dt.date().alias("date"),
                pl.col("quantity").cast(pl.Float64).alias("quantity"),
            )
            .group_by(["location", "item_id", "date"])
            .agg(pl.col("quantity").sum().alias("quantity"))
        )

    raise FileNotFoundError(
        f"Missing both cache {CACHE_PATH} and raw transaction file {TRANSACTIONS_PATH}"
    )


def add_date_features(lf):
    return lf.with_columns(
        pl.col("date").dt.weekday().alias("weekday"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.ordinal_day().alias("day_of_year"),
    )


def dataset_overview(daily_lf):
    overview = daily_lf.select(
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.len().alias("nonzero_pair_days"),
        pl.struct(["location", "item_id"]).n_unique().alias("pairs"),
        pl.col("location").n_unique().alias("locations"),
        pl.col("item_id").n_unique().alias("items"),
        pl.col("quantity").sum().alias("total_quantity"),
        pl.col("quantity").mean().alias("mean_quantity_on_sold_day"),
        pl.col("quantity").median().alias("median_quantity_on_sold_day"),
        pl.col("quantity").quantile(0.90).alias("p90_quantity_on_sold_day"),
        pl.col("quantity").quantile(0.99).alias("p99_quantity_on_sold_day"),
        pl.col("quantity").max().alias("max_quantity_on_sold_day"),
    ).collect()
    min_date, max_date = overview.select("min_date", "max_date").row(0)
    n_days = (max_date - min_date).days + 1
    return overview.with_columns(
        pl.lit(n_days).alias("calendar_days"),
        (pl.col("nonzero_pair_days") / (pl.col("pairs") * n_days)).alias("global_active_rate"),
    )


def pair_summary(daily_lf):
    base = (
        daily_lf.group_by(["location", "item_id"])
        .agg(
            pl.col("date").min().alias("first_sold_date"),
            pl.col("date").max().alias("last_sold_date"),
            pl.col("date").n_unique().alias("active_days"),
            pl.col("quantity").sum().alias("total_quantity"),
            pl.col("quantity").mean().alias("mean_quantity_on_sold_day"),
            pl.col("quantity").median().alias("median_quantity_on_sold_day"),
            pl.col("quantity").std().fill_null(0).alias("std_quantity_on_sold_day"),
            pl.col("quantity").quantile(0.25).alias("q25_quantity_on_sold_day"),
            pl.col("quantity").quantile(0.75).alias("q75_quantity_on_sold_day"),
            pl.col("quantity").quantile(0.90).alias("p90_quantity_on_sold_day"),
            pl.col("quantity").quantile(0.99).alias("p99_quantity_on_sold_day"),
            pl.col("quantity").max().alias("max_quantity_on_sold_day"),
        )
        .with_columns(
            (
                (pl.col("last_sold_date") - pl.col("first_sold_date")).dt.total_days() + 1
            ).alias("observed_span_days"),
        )
        .with_columns(
            (pl.col("active_days") / pl.col("observed_span_days")).alias("active_rate_in_span"),
            (
                pl.col("std_quantity_on_sold_day")
                / pl.col("mean_quantity_on_sold_day").clip(lower_bound=1e-9)
            ).alias("cv_quantity_on_sold_day"),
            (
                pl.col("max_quantity_on_sold_day")
                / pl.col("median_quantity_on_sold_day").clip(lower_bound=1e-9)
            ).alias("max_to_median_ratio"),
        )
    )

    gaps = (
        daily_lf.sort(["location", "item_id", "date"])
        .with_columns(
            (
                pl.col("date").diff().over(["location", "item_id"]).dt.total_days()
            ).alias("gap_days")
        )
        .filter(pl.col("gap_days").is_not_null())
        .group_by(["location", "item_id"])
        .agg(
            pl.col("gap_days").mean().alias("mean_gap_between_sold_days"),
            pl.col("gap_days").median().alias("median_gap_between_sold_days"),
            pl.col("gap_days").max().alias("max_gap_between_sold_days"),
        )
    )

    return base.join(gaps, on=["location", "item_id"], how="left").with_columns(
        pl.col("mean_gap_between_sold_days").fill_null(0),
        pl.col("median_gap_between_sold_days").fill_null(0),
        pl.col("max_gap_between_sold_days").fill_null(0),
    )


def outlier_days(daily_lf, pair_stats_lf):
    thresholds = pair_stats_lf.select(
        "location",
        "item_id",
        (
            pl.col("q75_quantity_on_sold_day")
            + 1.5 * (pl.col("q75_quantity_on_sold_day") - pl.col("q25_quantity_on_sold_day"))
        ).alias("high_outlier_threshold"),
    )
    return (
        daily_lf.join(thresholds, on=["location", "item_id"], how="inner")
        .filter(pl.col("quantity") > pl.col("high_outlier_threshold"))
        .select(["location", "item_id", "date", "quantity", "high_outlier_threshold"])
        .sort("quantity", descending=True)
    )


def write_table(df_or_lf, path, top_n=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df_or_lf.collect(streaming=True) if isinstance(df_or_lf, pl.LazyFrame) else df_or_lf
    if top_n is not None:
        df = df.head(top_n)
    df.write_csv(path)
    log(f"Wrote {path} ({df.height:,} rows)")
    return df


def write_json_summary(overview_df, path):
    row = overview_df.to_dicts()[0]
    serializable = {key: str(value) if hasattr(value, "isoformat") else value for key, value in row.items()}
    Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    log(f"Wrote {path}")


def save_bar(df, x, y, title, path, rotation=45):
    if df.height == 0:
        return
    plt.figure(figsize=(12, 6))
    plt.bar(df[x].cast(str).to_list(), df[y].to_list())
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    log(f"Wrote {path}")


def save_line(df, x, y, title, path):
    if df.height == 0:
        return
    plt.figure(figsize=(14, 6))
    plt.plot(df[x].to_list(), df[y].to_list(), linewidth=1.2)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    log(f"Wrote {path}")


def save_hist(values, title, path, bins=80, log_y=True):
    values = [v for v in values if v is not None]
    if not values:
        return
    plt.figure(figsize=(12, 6))
    plt.hist(values, bins=bins)
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    log(f"Wrote {path}")


def save_scatter(df, x, y, title, path):
    if df.height == 0:
        return
    sample = df.head(50000)
    plt.figure(figsize=(10, 7))
    plt.scatter(sample[x].to_list(), sample[y].to_list(), s=4, alpha=0.25)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    log(f"Wrote {path}")


def generate_plots(daily_lf, pair_stats_df):
    daily_totals = (
        daily_lf.group_by("date")
        .agg(
            pl.col("quantity").sum().alias("total_quantity"),
            pl.len().alias("sold_pair_days"),
        )
        .sort("date")
        .collect(streaming=True)
    )
    save_line(
        daily_totals,
        "date",
        "total_quantity",
        "Total sold quantity by day, excluding zero pair-days",
        EDA_DIR / "daily_total_quantity.png",
    )
    save_line(
        daily_totals,
        "date",
        "sold_pair_days",
        "Number of active location-item pairs by day",
        EDA_DIR / "daily_active_pairs.png",
    )

    weekday = (
        add_date_features(daily_lf)
        .group_by("weekday")
        .agg(pl.col("quantity").sum().alias("total_quantity"))
        .sort("weekday")
        .collect(streaming=True)
    )
    save_bar(weekday, "weekday", "total_quantity", "Total quantity by weekday", EDA_DIR / "weekday_quantity.png", 0)

    top_locations = (
        daily_lf.group_by("location")
        .agg(pl.col("quantity").sum().alias("total_quantity"))
        .sort("total_quantity", descending=True)
        .limit(30)
        .collect(streaming=True)
    )
    save_bar(
        top_locations,
        "location",
        "total_quantity",
        "Top 30 locations by sold quantity",
        EDA_DIR / "top_locations_quantity.png",
    )

    top_items = (
        daily_lf.group_by("item_id")
        .agg(pl.col("quantity").sum().alias("total_quantity"))
        .sort("total_quantity", descending=True)
        .limit(30)
        .collect(streaming=True)
    )
    save_bar(top_items, "item_id", "total_quantity", "Top 30 items by sold quantity", EDA_DIR / "top_items_quantity.png")

    quantity_sample = daily_lf.select("quantity").limit(2_000_000).collect(streaming=True)
    save_hist(
        quantity_sample["quantity"].to_list(),
        "Sold-day quantity distribution, zeros excluded",
        EDA_DIR / "sold_day_quantity_hist.png",
    )
    save_hist(
        pair_stats_df["active_days"].to_list(),
        "Active sold-day count per location-item pair",
        EDA_DIR / "pair_active_days_hist.png",
    )
    save_scatter(
        pair_stats_df.sort("total_quantity", descending=True),
        "active_days",
        "total_quantity",
        "Pair active days vs total quantity, zeros excluded",
        EDA_DIR / "pair_active_days_vs_total_quantity.png",
    )


def focus_pair(daily_lf, location, item_id):
    pair_df = (
        daily_lf.filter((pl.col("location") == location) & (pl.col("item_id") == str(item_id)))
        .sort("date")
        .collect(streaming=True)
    )
    if pair_df.height == 0:
        log(f"No non-zero daily sales found for location={location}, item_id={item_id}")
        return

    pair_path = EDA_DIR / f"pair_location_{location}_item_{item_id}.csv"
    pair_df.write_csv(pair_path)
    log(f"Wrote {pair_path} ({pair_df.height:,} rows)")

    save_line(
        pair_df,
        "date",
        "quantity",
        f"Daily quantity for location={location}, item_id={item_id}; zeros excluded",
        EDA_DIR / f"pair_location_{location}_item_{item_id}.png",
    )


def main():
    parser = argparse.ArgumentParser(description="EDA for daily quantity by (location, item_id).")
    parser.add_argument("--location", type=int, default=None, help="Optional location to inspect.")
    parser.add_argument("--item-id", default=None, help="Optional item_id to inspect.")
    parser.add_argument("--top-n", type=int, default=1000, help="Rows to export for ranked CSV tables.")
    args = parser.parse_args()

    ensure_output_dirs()
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    daily_lf = scan_daily_quantity().filter(pl.col("quantity") > 0)
    overview = dataset_overview(daily_lf)
    write_table(overview, EDA_DIR / "overview.csv")
    write_json_summary(overview, EDA_DIR / "overview.json")

    pair_stats_lf = pair_summary(daily_lf)
    pair_stats_df = pair_stats_lf.collect(streaming=True)
    pair_stats_df.write_parquet(EDA_DIR / "pair_summary.parquet")
    log(f"Wrote {EDA_DIR / 'pair_summary.parquet'} ({pair_stats_df.height:,} rows)")

    write_table(
        pair_stats_df.sort("total_quantity", descending=True),
        EDA_DIR / "top_pairs_by_total_quantity.csv",
        args.top_n,
    )
    write_table(
        pair_stats_df.sort("active_days", descending=True),
        EDA_DIR / "top_pairs_by_active_days.csv",
        args.top_n,
    )
    write_table(
        pair_stats_df.filter(pl.col("active_days") >= 10).sort("cv_quantity_on_sold_day", descending=True),
        EDA_DIR / "volatile_pairs_min_10_active_days.csv",
        args.top_n,
    )
    write_table(
        pair_stats_df.sort("max_gap_between_sold_days", descending=True),
        EDA_DIR / "intermittent_pairs_by_max_gap.csv",
        args.top_n,
    )
    write_table(
        outlier_days(daily_lf, pair_stats_lf),
        EDA_DIR / "high_outlier_sold_days.csv",
        args.top_n,
    )

    location_summary = (
        daily_lf.group_by("location")
        .agg(
            pl.col("quantity").sum().alias("total_quantity"),
            pl.len().alias("nonzero_pair_days"),
            pl.col("item_id").n_unique().alias("items_sold"),
            pl.struct(["location", "item_id"]).n_unique().alias("pairs_sold"),
        )
        .sort("total_quantity", descending=True)
    )
    write_table(location_summary, EDA_DIR / "location_summary.csv")

    item_summary = (
        daily_lf.group_by("item_id")
        .agg(
            pl.col("quantity").sum().alias("total_quantity"),
            pl.len().alias("nonzero_pair_days"),
            pl.col("location").n_unique().alias("locations_sold"),
            pl.struct(["location", "item_id"]).n_unique().alias("pairs_sold"),
        )
        .sort("total_quantity", descending=True)
    )
    write_table(item_summary, EDA_DIR / "item_summary.csv")

    date_summary = (
        add_date_features(daily_lf)
        .group_by("date")
        .agg(
            pl.col("quantity").sum().alias("total_quantity"),
            pl.len().alias("nonzero_pair_days"),
            pl.col("location").n_unique().alias("active_locations"),
            pl.col("item_id").n_unique().alias("active_items"),
        )
        .sort("date")
    )
    write_table(date_summary, EDA_DIR / "date_summary.csv")

    weekday_summary = (
        add_date_features(daily_lf)
        .group_by("weekday")
        .agg(
            pl.col("quantity").sum().alias("total_quantity"),
            pl.col("quantity").mean().alias("mean_quantity_on_sold_day"),
            pl.len().alias("nonzero_pair_days"),
        )
        .sort("weekday")
    )
    write_table(weekday_summary, EDA_DIR / "weekday_summary.csv")

    generate_plots(daily_lf, pair_stats_df)

    if args.location is not None and args.item_id is not None:
        focus_pair(daily_lf, args.location, args.item_id)
    elif args.location is not None or args.item_id is not None:
        log("Provide both --location and --item-id to export one pair's daily series.")

    log(f"EDA complete. Open files under {EDA_DIR}")


if __name__ == "__main__":
    main()