import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = Path(os.getenv("SALES_FORECASTING_ROOT", PROJECT_DIR))
RAW_DATA_DIR = Path(os.getenv("SALES_FORECASTING_DATA_DIR", ROOT_DIR / "data"))
OUTPUT_DIR = Path(os.getenv("SALES_FORECASTING_OUTPUT_DIR", ROOT_DIR / "outputs"))

# Đường dẫn 3 file parquet
TRANSACTIONS_PATH = RAW_DATA_DIR / "transaction_full_2025.parquet"
ITEMS_PATH = RAW_DATA_DIR / "items.parquet"
EVENT_PATH = RAW_DATA_DIR / "event_full_2025.parquet"

CACHE_DIR = OUTPUT_DIR / "cache"
LOG_DIR = OUTPUT_DIR / "logs"

def ensure_output_dirs() -> None:
    for path in [CACHE_DIR, LOG_DIR, OUTPUT_DIR / "eda"]:
        path.mkdir(parents=True, exist_ok=True)

def first_existing(*paths: str | Path) -> Path:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return Path(paths[0])
