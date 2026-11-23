# Feature analysis

The forecasting target is monthly `quantity` for each `(location, item_id)` pair. Features are computed using data available before the target month so that the model does not see future sales.

## Feature groups

| Group | Examples | Why it is useful |
| --- | --- | --- |
| Pair history | `pair_qty_all`, `pair_qty_30d`, `pair_qty_90d`, `pair_qty_180d`, `pair_qty_365d`, active days, days since last sale | Captures the specific location-item demand pattern and recency. |
| Item history | Rolling quantity and active days for `item_id` | Captures an item's overall popularity across locations. |
| Location history | Rolling quantity and active days for `location` | Captures store or location scale and recent traffic. |
| Category history | Aggregates for `category_l1` and `category_l2` | Shares signal across related products when a pair has sparse history. |
| Brand history | Rolling aggregates for `brand` | Captures brand-level demand and recognition. |
| Location-category history | Aggregates for `(location, category_l1)` | Captures local preferences by product group. |
| Item metadata | `price`, category fields, `brand`, `manufacturer`, `size` | Adds stable product information and categorical context. |
| Derived ratios | Recent location share, pair-to-location share, pair-to-item share | Normalizes demand by the scale of the location or item. |
| Calendar context | Target month and target-month day count | Lets the model adjust for month length and seasonality. |

## Transformations

- Transactions are filtered to purchase events and aggregated to daily quantity by location, item, and date.
- The rolling windows are 30, 60, 90, 180, and 365 days.
- Missing numeric aggregates are filled with zero; missing categorical values become `unknown`.
- Last-sale timestamps are converted to `days_since_sale` relative to the feature cutoff.
- Monthly targets are normalized for excluded anomaly days by scaling valid-day totals to the full calendar month.

## Model input and target handling

The model receives 109 columns in the recorded run. Categorical columns include location, item, category, brand, manufacturer, and size. The target is split into:

1. a binary positive-sale label (`quantity > 0`) for `CatBoostClassifier`;
2. the raw positive quantity for `CatBoostRegressor` with MAPE loss.

The final prediction multiplies the classifier probability by the positive-quantity estimate, applies the validation-tuned scale, rounds to an integer, and clips negative values to zero.
