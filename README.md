# Sales Forecasting Pipeline

This repository contains the notebook exported from Kaggle:

- Kaggle: https://www.kaggle.com/code/thnhhuytrn/sale-forecasting
- Notebook: [`sale_forecasting.ipynb`](./sale_forecasting.ipynb)

The notebook reconstructs the forecasting pipeline in the Kaggle working directory, including feature preparation, CatBoost classification/regression, December validation, January 2026 prediction, EDA outputs, and submission pickle cleanup.

## Kaggle data

The notebook expects the private Kaggle dataset `thnhhuytrn/saleforecasting` with these files:

- `transaction_full_2025.parquet`
- `items.parquet`
- `event_full_2025.parquet`

The raw data is intentionally excluded from this repository. Configure the dataset in Kaggle or update the paths in the generated scripts before running locally.

## Main result

The recorded Kaggle run reports a December validation score of approximately `60.27` and writes a fixed January submission with `7,110,300` rows.

## Run

Open `sale_forecasting.ipynb` in Kaggle or Jupyter and execute the cells in order. The notebook writes the pipeline scripts into its working directory before running them.

## License

The Kaggle notebook declares the Apache 2.0 license.
