# Review resolution

This document tracks the external AI/ML intern portfolio review against the repository changes.

## Confirmed strengths

- Realistic monthly location-item demand forecasting over tens of millions of transactions.
- Polars lazy scans, streaming aggregation, and a reusable daily cache.
- Multi-level rolling, recency, categorical, and demand-share features.
- Time-based cutoff and December holdout design.
- Committed EDA charts, tables, feature documentation, and privacy-aware data handling.

## Completed fixes

| Review finding | Resolution | Evidence |
| --- | --- | --- |
| Negative labels used only sampled positive keys | `build_training_data` now passes all observed positive keys to `negative_sample` | `src/sales_forecasting/train_predict.py` |
| Negative chunks could contain duplicate keys | Sampling anti-joins accumulated keys, caps to the available universe, and asserts zero positive overlap | `tests/test_sampling.py` |
| Final inference was described as probability × amount | Documentation now states that the recorded `proba_power=0` configuration behaves as a scaled regressor | `docs/model_card.md`, `docs/feature_analysis.md` |
| Training code existed only inside notebook cells | Pipeline modules are versioned under `src/sales_forecasting` | `src/sales_forecasting/` |
| Hard-coded runtime paths | Data, root, and output paths now accept environment variables with local defaults | `src/sales_forecasting/project_paths.py` |
| No reproducible public smoke data | Deterministic synthetic Parquet generator added | `scripts/generate_sample_data.py` |
| No regression tests or CI | Sampling and metric tests plus a synthetic EDA workflow run in GitHub Actions | `tests/`, `.github/workflows/tests.yml` |
| Broad dependency ranges | Portfolio environment versions are pinned | `requirements.txt`, `pyproject.toml` |
| Partial license text | Replaced with the complete Apache 2.0 terms | `LICENSE` |
| Missing model handoff documentation | Added model card with intended use, selected config, metrics, and limitations | `docs/model_card.md` |

## Evidence that must be regenerated

The notebook metrics `60.2692` and `60.2592` were produced before the negative-sampling correction. They remain historical evidence only. A full retraining run is required before claiming an updated score for the corrected source.

## Open validation work

- Verify whether item metadata is point-in-time correct or a final-year snapshot.
- Run rolling-origin backtests across multiple target months.
- Compare anomaly handling with no removal, removal without scaling, and removal with scaling.
- Add feature importance or SHAP after retraining the corrected model.
- Report full-grid MAE, WMAPE, occurrence precision/recall, and overstock versus understock error slices.
- Keep CV claims aligned with the public implementation: multi-window aggregates, recency, metadata, and demand-share features; do not claim velocity, acceleration, lifecycle, or ranking features without code evidence.
