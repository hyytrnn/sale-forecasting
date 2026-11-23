"""Generate a small deterministic dataset with the production input schema."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("sample-data"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    item_ids = [f"ITEM-{index:03d}" for index in range(1, 9)]
    locations = ["101", "102", "103"]
    items = pl.DataFrame(
        {
            "item_id": item_ids,
            "price": [float(10 + index * 3) for index in range(len(item_ids))],
            "category_l1": ["food"] * 4 + ["household"] * 4,
            "category_l2": ["snack", "snack", "drink", "drink", "cleaning", "cleaning", "paper", "paper"],
            "category_l3": ["sample"] * len(item_ids),
            "category": ["synthetic"] * len(item_ids),
            "brand": [f"BRAND-{index % 3}" for index in range(len(item_ids))],
            "manufacturer": [f"MFG-{index % 2}" for index in range(len(item_ids))],
            "description": ["Synthetic portfolio sample"] * len(item_ids),
            "sale_status": [1] * len(item_ids),
            "size": ["standard"] * len(item_ids),
        }
    )
    rows: list[dict[str, object]] = []
    start = datetime(2025, 1, 1)
    bill_id = 1
    for day_offset in range(365):
        date = start + timedelta(days=day_offset)
        for location in locations:
            for item_index, item_id in enumerate(item_ids):
                seasonal = 1.0 + 0.35 * np.sin(2 * np.pi * day_offset / 30)
                demand = max(0.0, seasonal * (item_index + 1) / 4)
                quantity = int(rng.poisson(demand))
                if quantity == 0:
                    continue
                rows.append(
                    {
                        "customer_id": int(rng.integers(1, 501)),
                        "item_id": item_id,
                        "price": float(items["price"][item_index]),
                        "location": int(location),
                        "discount": float(rng.choice([0.0, 0.05, 0.10])),
                        "bill_id": bill_id,
                        "quantity": quantity,
                        "event_type": "Purchase",
                        "updated_date": date,
                    }
                )
                bill_id += 1
    transactions = pl.DataFrame(rows)
    events = transactions.select(
        "customer_id",
        "item_id",
        "price",
        "quantity",
        "event_type",
        pl.col("updated_date").alias("event_date"),
        pl.col("updated_date").alias("created_date"),
        "updated_date",
    )
    items.write_parquet(args.output_dir / "items.parquet")
    transactions.write_parquet(args.output_dir / "transaction_full_2025.parquet")
    events.write_parquet(args.output_dir / "event_full_2025.parquet")
    print(f"Wrote {transactions.height:,} synthetic transactions to {args.output_dir}")


if __name__ == "__main__":
    main()
