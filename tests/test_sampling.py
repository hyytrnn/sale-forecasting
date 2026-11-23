import polars as pl
from types import SimpleNamespace

from sales_forecasting.train_predict import negative_sample, sample_training_pairs


def test_negative_sample_is_unique_and_disjoint_from_all_positives() -> None:
    locations = pl.DataFrame({"location": ["1", "2"]})
    items = pl.DataFrame({"item_id": ["a", "b", "c"]})
    all_positives = pl.DataFrame(
        {"location": ["1", "2"], "item_id": ["a", "b"], "quantity": [2.0, 3.0]}
    )

    negatives = negative_sample(locations, items, all_positives, n_rows=4, seed=42)

    assert negatives.height == 4
    assert negatives.select(["location", "item_id"]).unique().height == 4
    overlap = negatives.join(all_positives, on=["location", "item_id"], how="inner")
    assert overlap.height == 0
    assert negatives["quantity"].to_list() == [0.0] * 4


def test_negative_sample_caps_at_available_candidate_universe() -> None:
    locations = pl.DataFrame({"location": ["1"]})
    items = pl.DataFrame({"item_id": ["a", "b"]})
    all_positives = pl.DataFrame(
        {"location": ["1"], "item_id": ["a"], "quantity": [1.0]}
    )

    negatives = negative_sample(locations, items, all_positives, n_rows=10, seed=7)

    assert negatives.select(["location", "item_id"]).to_dicts() == [
        {"location": "1", "item_id": "b"}
    ]


def test_unsampled_positive_is_never_relabeled_as_negative() -> None:
    locations = pl.DataFrame({"location": ["1", "2"]})
    items = pl.DataFrame({"item_id": ["a", "b", "c"]})
    target = pl.DataFrame(
        {
            "location": ["1", "2"],
            "item_id": ["a", "b"],
            "quantity": [2.0, 3.0],
        }
    )
    args = SimpleNamespace(
        max_positive_per_month=1,
        max_negative_per_month=4,
        negative_ratio=4,
    )

    positives, negatives = sample_training_pairs(target, locations, items, args, month=4)

    assert positives.height == 1
    assert negatives.height == 4
    overlap = negatives.join(target, on=["location", "item_id"], how="inner")
    assert overlap.height == 0
