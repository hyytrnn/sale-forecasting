"""Materialize the Python modules embedded in the project notebook.

The notebook remains a historical experiment record. Versioned modules under
``src/sales_forecasting`` are the maintainable source used by tests and CI.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "sale_forecasting.ipynb"
OUTPUT_DIR = ROOT / "src" / "sales_forecasting"
ALLOWED = {
    "project_paths.py",
    "train_predict.py",
    "train_predict_decval.py",
    "train_predict_decval_v2.py",
    "prepare_cache.py",
    "make_subset.py",
    "evaluator.py",
    "fix_submission_pickle.py",
    "eda.py",
}


def package_imports(filename: str, source: str) -> str:
    replacements = {
        "from project_paths import ": "from .project_paths import ",
        "import train_predict as tp": "from . import train_predict as tp",
        "from train_predict_decval import main": "from .train_predict_decval import main",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite versioned modules with notebook cells. Review fixes afterward.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        first_line, separator, body = source.partition("\n")
        if not first_line.startswith("%%writefile "):
            continue
        filename = first_line.removeprefix("%%writefile ").strip()
        if filename not in ALLOWED:
            continue
        materialized = package_imports(filename, body if separator else "")
        destination = OUTPUT_DIR / filename
        if destination.exists() and not args.force:
            continue
        destination.write_text(materialized, encoding="utf-8")
        written.append(filename)
    init_path = OUTPUT_DIR / "__init__.py"
    if not init_path.exists() or args.force:
        init_path.write_text(
            '"""Sales forecasting training and evaluation pipeline."""\n',
            encoding="utf-8",
        )
    print(f"Materialized {len(written)} modules in {OUTPUT_DIR}")
    print("\n".join(sorted(written)))


if __name__ == "__main__":
    main()
