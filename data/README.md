# Local data

The three Parquet files in this directory are local inputs and are intentionally excluded from Git:

- `transaction_full_2025.parquet`: transaction events with location, item, date, event type, and quantity;
- `items.parquet`: item metadata and sale status;
- `event_full_2025.parquet`: event-level data used by the subset utility.

The files contain transaction-level records, including `customer_id`, and total about 949 MB. Keep them in local storage or a private data registry. Do not publish them in the public GitHub repository without a separate privacy and licensing review.

Run `python scripts/run_eda.py` from the project root to build `daily_quantity.parquet` and generate the tracked EDA artifacts under `reports/eda/`.

For a public smoke test without the private files, run `python scripts/generate_sample_data.py --output-dir sample-data` and point the EDA command at `sample-data/`.
