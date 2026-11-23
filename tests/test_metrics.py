import numpy as np

from sales_forecasting.evaluator import zero_aware_metrics
from sales_forecasting.train_predict import score_metric


def test_score_metric_ignores_zero_actuals_by_benchmark_definition() -> None:
    actual = np.array([10.0, 20.0, 0.0])
    predicted = np.array([8.0, 22.0, 999.0])

    assert score_metric(actual, predicted) == 85.0


def test_score_metric_returns_zero_when_no_positive_actual_exists() -> None:
    assert score_metric(np.zeros(3), np.ones(3)) == 0.0


def test_zero_aware_metrics_penalize_false_positive_demand() -> None:
    actual = np.array([10.0, 0.0, 0.0, 5.0])
    predicted = np.array([8.0, 4.0, 0.0, 5.0])

    metrics = zero_aware_metrics(actual, predicted)

    assert metrics["mae"] == 1.5
    assert metrics["wmape"] == 0.4
    assert metrics["positive_precision"] == 2 / 3
    assert metrics["positive_recall"] == 1.0
    assert metrics["predicted_positive_rate"] == 0.75
