import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/runs/decval_v2_scaled_max5/january_2026/submission.pkl")
DEFAULT_OUTPUT = Path("submission.pkl")


def normalize_submission(input_path: Path, output_path: Path) -> None:
    with input_path.open("rb") as f:
        submission = pickle.load(f)

    df = pd.DataFrame(submission)
    missing = {"location", "item_id", "quantity"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    normalized = pd.DataFrame(
        {
            "location": pd.to_numeric(df["location"], downcast="integer").to_numpy(dtype=np.int32),
            "item_id": df["item_id"].astype(str).to_numpy(dtype=object),
            "quantity": df["quantity"].fillna(0).round().clip(lower=0).to_numpy(dtype=np.int64),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(normalized, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {output_path}")
    print(f"shape={normalized.shape}")
    print(normalized.dtypes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite submission.pkl to match evaluator dtypes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Overwrite every outputs/runs/*/*/submission.pkl in place.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all_runs:
        paths = sorted(Path("outputs/runs").glob("*/*/submission.pkl"))
        if not paths:
            raise SystemExit("No outputs/runs/*/*/submission.pkl files found.")
        for path in paths:
            normalize_submission(path, path)
    else:
        normalize_submission(args.input, args.output)