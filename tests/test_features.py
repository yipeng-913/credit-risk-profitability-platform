import numpy as np
import pytest
from src.features.build import build_features, FEATURES
from src.data.validation import validate_frames


def test_cardinality_and_allowlist(frames):
    result = build_features(frames)
    assert result.SK_ID_CURR.is_unique
    assert len(result) == len(frames["application_train"])
    assert result.columns.tolist() == ["SK_ID_CURR", "TARGET"] + FEATURES
    assert "TARGET" not in FEATURES and "SK_ID_CURR" not in FEATURES


def test_historical_aggregation_and_future_filter(frames):
    key = frames["application_train"].SK_ID_CURR.iloc[0]
    b = frames["bureau"].iloc[:3].copy()
    b["SK_ID_CURR"] = key
    b["DAYS_CREDIT"] = [-10, -20, 10]
    b["AMT_CREDIT_SUM"] = [100, 200, 999999]
    frames["bureau"] = b
    result = build_features(frames).set_index("SK_ID_CURR")
    assert result.loc[key, "bureau_count"] == 2
    assert result.loc[key, "bureau_credit_sum"] == 300
    other = result.drop(index=key)
    assert other.bureau_count.eq(0).all()
    assert other.no_bureau_history.eq(1).all()


def test_duplicate_rejected(frames):
    frames["application_train"].loc[1, "SK_ID_CURR"] = frames["application_train"].SK_ID_CURR.iloc[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_frames(frames)


@pytest.mark.parametrize("value", [2, np.nan, -1])
def test_target_domain(frames, value):
    frames["application_train"].loc[0, "TARGET"] = value
    with pytest.raises(ValueError, match="TARGET"):
        validate_frames(frames)


def test_sentinel_and_division_policy(frames):
    frames["application_train"].loc[0, ["DAYS_EMPLOYED", "AMT_INCOME_TOTAL"]] = [365243, 0]
    result = build_features(frames)
    assert np.isnan(result.loc[0, "employment_years"])
    assert np.isnan(result.loc[0, "credit_income_ratio"])
    assert result.loc[0, "employment_missing"] == 1


def test_target_cannot_influence_features(frames):
    a = build_features(frames)
    frames["application_train"]["TARGET"] = 1 - frames["application_train"].TARGET
    b = build_features(frames)
    assert a[FEATURES].equals(b[FEATURES])
