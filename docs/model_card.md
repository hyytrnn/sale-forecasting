# Model card

## Intended use

The project forecasts monthly retail quantity for every candidate `(location, item_id)` pair. It is a portfolio and benchmark-oriented implementation, not a deployed inventory decision system.

## Inputs and target

- Purchase transactions aggregated to daily quantity.
- Item metadata including price, categories, brand, manufacturer, size, and `sale_status`.
- Monthly quantity is the regression target; `quantity > 0` is the optional occurrence-classification target.

## Validation design

Training examples use only dates available before each target month. The recorded experiment trains through November 2025, tunes on December 2025, retrains through December, and predicts January 2026.

## Experimental architecture

The code supports a hurdle-style model: a CatBoost classifier estimates positive-sale probability and a CatBoost regressor estimates positive quantity. Probability modulation, thresholds, baseline blending, and quantity-bucket blending are tunable.

## Selected recorded configuration

The recorded validation selected `threshold=0.0`, `scale=0.67`, `blend=0.0`, `bucket_blend=0.0`, and `proba_power=0.0`. With `proba_power=0.0` and `threshold=0.0`, classifier probability does not affect the selected final quantity prediction. The effective recorded formula is approximately:

```text
prediction = round(max(0, regressor_quantity) * 0.67)
```

The classifier remains part of the experimental architecture, but the selected benchmark configuration behaves as a scaled positive-quantity regressor.

## Metrics and recorded evidence

| Result | Metric | Evaluation set | Status |
| --- | --- | --- | --- |
| 60.2692 | `100 - nonzero MAPE%` | December tuning sample | Historical notebook output before the sampling fix |
| 60.2592 | `100 - nonzero MAPE%` | December full-grid evaluation | Historical notebook output before the sampling fix |

These values are not classification accuracy. Corrected sampling code must be retrained before reporting a new comparable score.

## Known limitations

- The benchmark metric ignores ground-truth zero rows, so false-positive demand is not directly penalized.
- The recorded final configuration predicts positive demand for nearly the entire January candidate grid.
- Item metadata may be a final snapshot rather than point-in-time history; this must be verified before claiming complete leakage prevention.
- Validation currently centers on one holdout month rather than rolling-origin folds.
- Removing anomaly days and scaling to calendar days needs an ablation study.
- Candidate items come from global `sale_status == 1`, not location-specific availability.

## Production-oriented evaluation to add

Use MAE and WMAPE over the complete grid, positive-occurrence precision/recall, overstock versus understock cost, rolling-origin backtests, and error slices for sparse pairs, cold-start items, locations, categories, and demand buckets.
