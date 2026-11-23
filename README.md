# Sales Forecasting Pipeline

This repository contains a reproducible sales forecasting notebook and the Python modules that the notebook writes before executing the pipeline. The task is to estimate the monthly quantity sold for every `(location, item_id)` pair.

The main notebook is [`sale_forecasting.ipynb`](./sale_forecasting.ipynb). It is organized as a set of code-generation cells followed by two execution cells.

## Project structure

The notebook creates these modules in the working directory:

| Module | Purpose |
| --- | --- |
| `project_paths.py` | Centralizes raw-data, cache, log, model, and output paths. |
| `train_predict.py` | Builds the daily sales cache, creates features, trains models, scores predictions, and provides prediction strategies. |
| `train_predict_decval.py` | Runs month-based training, December validation/tuning, full-grid prediction, retraining, and artifact export. |
| `train_predict_decval_v2.py` | Thin wrapper that injects the final validation and regressor defaults before calling the main pipeline. |
| `prepare_cache.py` | Builds `daily_quantity.parquet` with a streaming Polars aggregation. |
| `make_subset.py` | Creates a smaller data subset for local smoke tests. |
| `eda.py` | Produces descriptive tables, outlier reports, rankings, and plots. |
| `evaluator.py` | Compares a prediction pickle with an answer pickle using the project score. |
| `fix_submission_pickle.py` | Normalizes submission columns and dtypes before delivery. |

The notebook itself is the source of these modules; they are not duplicated as separate files in this repository.

## Data used by the pipeline

The pipeline expects three Parquet inputs:

- `transaction_full_2025.parquet`: transaction-level sales events.
- `items.parquet`: item metadata and active-sale status.
- `event_full_2025.parquet`: optional event data used when creating a subset.

The transaction data must provide at least `location`, `item_id`, `updated_date`, `event_type`, and `quantity`. The item table supplies `item_id`, `sale_status`, `price`, `category_l1`, `category_l2`, `category_l3`, `category`, `brand`, `manufacturer`, and `size`.

Raw data is not committed to this repository. Set the paths in `project_paths.py` to the location of the Parquet files before running locally. The pipeline filters `event_type == "Purchase"`, casts IDs to strings, converts timestamps to calendar dates, and aggregates quantity by `(location, item_id, date)`.

The recorded run found 1,038 locations, 6,850 active items, and 7,110,300 possible location-item pairs. Seven known anomalous dates are excluded from training and evaluation:

`2025-01-29`, `2025-01-30`, `2025-01-31`, `2025-07-04`, `2025-07-18`, `2025-09-29`, and `2025-10-27`.

## Exploratory data analysis

`eda.py` reads the daily quantity table and keeps non-zero sold pair-days. It produces:

- an overview containing the date range, number of locations, items, pairs, total quantity, mean, median, P90, P99, and maximum quantity;
- pair-level summaries with first/last sale dates, active days, total quantity, dispersion, coefficient of variation, and gaps between sales;
- IQR-based high-outlier days for each pair;
- ranked CSV tables for pairs, locations, items, dates, weekdays, volatile pairs, and intermittent pairs;
- PNG plots for daily totals, active pairs, weekday totals, top locations, top items, quantity distribution, active-day distribution, and pair activity versus total quantity.

Outputs are written under `eda/`. The notebook defines this EDA module but the main training cell does not execute it automatically; run `python eda.py` after the daily cache exists when EDA artifacts are needed.

## Feature engineering

For each target month, features are calculated using data available on the previous day. The pipeline builds rolling aggregates over 30, 60, 90, 180, and 365 days at several levels:

- `(location, item_id)` pair;
- item;
- location;
- `category_l1` and `category_l2`;
- brand;
- `(location, category_l1)`.

Each level contributes quantity totals, active-day counts, and recency. Date recency is converted to `days_since_sale`; missing numeric values become zero and missing categorical values become `unknown`. Additional features include recent location shares and pair-to-location/item shares. The target month number and number of calendar days are also included.

## Model and training procedure

The final run uses a two-stage CatBoost approach:

1. **Positive-sale classifier** — `CatBoostClassifier` with Logloss and AUC. It predicts whether a location-item pair has positive quantity.
2. **Positive-quantity regressor** — `CatBoostRegressor` trained only on positive rows with raw quantity targets and MAPE loss.

Both models use categorical features such as location, item, category, brand, manufacturer, and size. The recorded run uses CPU training, seed `42`, depth `8`, learning rate `0.06`, 700 classifier iterations, and 900 regressor iterations. The optional quantity-bucket classifier is disabled in the final command.

Training data is built month by month from April through November 2025. Each month samples up to 120,000 positive pairs and 240,000 negative pairs, giving 2,880,000 training rows and 109 columns in the recorded run. Negative pairs are sampled from the location-item grid and anti-joined against observed positive pairs.

December positive rows are used as the validation set, capped at 300,000 rows. The project score is:

```text
score = 100 - mean(abs(actual - predicted) / abs(actual)) * 100
```

Only non-zero actual quantities contribute to this score. The validation search tunes the prediction scale and related CatBoost controls. In the recorded run the best December tuning score was `60.2692` with scale `0.67`.

After validation, the models are retrained through December 2025. January 2026 predictions are generated for the full location-item grid in batches of 40 locations. Predictions are rounded, clipped at zero, and serialized with columns `location`, `item_id`, and `quantity`.

## Notebook execution

Run the notebook cells in order:

1. Write the path configuration and pipeline modules.
2. Execute `train_predict_decval_v2.py` with CPU training, December validation, 700 classifier iterations, 900 regressor iterations, and 40-location prediction batches.
3. Run `fix_submission_pickle.py` to normalize the final submission.

For a smaller local test, first run `make_subset.py` and use its generated paths with reduced iteration and sample settings. For a full run, ensure the Parquet files are available and enough memory and storage exist for the daily cache, model files, validation report, and full-grid submission.

## Recorded outputs

The completed notebook run recorded:

- December score after the full-grid evaluation: `60.2592`;
- January submission rows: `7,110,300`;
- January non-zero predictions: `7,110,273`;
- January predicted quantity sum: `8,621,707`;
- normalized submission schema: `location` as `int32`, `item_id` as `object`, and `quantity` as `int64`.

The run writes model checkpoints, `validation_report.csv`, December and January submission pickles, and a JSON metrics report under the configured `runs/<run_name>/` directory. The final normalization step writes `submission_fixed.pkl`.

## License

The notebook declares the Apache 2.0 license.
