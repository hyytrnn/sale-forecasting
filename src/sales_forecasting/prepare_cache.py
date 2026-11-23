from pathlib import Path

import polars as pl

from .project_paths import CACHE_DIR, LOG_DIR, RAW_DATA_DIR, ensure_output_dirs

TRANSACTIONS_PATH = RAW_DATA_DIR / "transaction_full_2025.parquet"
CACHE_PATH = CACHE_DIR / "daily_quantity.parquet"
LOG_PATH = LOG_DIR / "prepare_quantity_cache_log.txt"


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")

    def write(self, message):
        print(message, flush=True)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def main():
    ensure_output_dirs()
    logger = Logger(LOG_PATH)
    try:
        cache_path = Path(CACHE_PATH)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        logger.write("Task: build daily quantity cache with Polars streaming")
        logger.write(f"Input: {TRANSACTIONS_PATH}")
        logger.write(f"Output: {CACHE_PATH}")
        logger.write("Streaming aggregate started...")

        (
            pl.scan_parquet(TRANSACTIONS_PATH)
            .filter(pl.col("event_type") == "Purchase")
            .select(
                "location",
                "item_id",
                pl.col("updated_date").dt.date().alias("date"),
                pl.col("quantity").cast(pl.Float32).alias("quantity"),
            )
            .group_by(["location", "item_id", "date"])
            .agg(pl.col("quantity").sum().alias("y"))
            .sink_parquet(CACHE_PATH)
        )

        logger.write(f"Saved daily quantity cache to {CACHE_PATH}")
        logger.write(f"Saved prepare log to {LOG_PATH}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()