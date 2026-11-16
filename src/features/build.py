"""Strict allowlist; aggregate before joining, and retain missing-history indicators."""

import numpy as np
from src.data.validation import validate_frames

FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "EXT_SOURCE_2",
    "NAME_INCOME_TYPE",
    "age_years",
    "employment_years",
    "employment_missing",
    "credit_income_ratio",
    "annuity_income_ratio",
    "bureau_count",
    "bureau_credit_sum",
    "bureau_debt_sum",
    "bureau_overdue_max",
    "bureau_days_mean",
    "previous_count",
    "previous_amount_mean",
    "previous_refusal_rate",
    "no_bureau_history",
    "no_previous_history",
]


def ratio(a, b):
    return a / b.where(b > 0)


def build_features(frames):
    validate_frames(frames)
    a = frames["application_train"].copy()
    for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]:
        a[c] = a[c].where(a[c] > 0)
    a["EXT_SOURCE_2"] = a.EXT_SOURCE_2.where(a.EXT_SOURCE_2.between(0, 1))
    a["age_years"] = (-a.DAYS_BIRTH / 365.25).where((-a.DAYS_BIRTH / 365.25).between(18, 100))
    employment = a.DAYS_EMPLOYED.where(a.DAYS_EMPLOYED.le(0))
    a["employment_missing"] = employment.isna().astype(int)
    a["employment_years"] = -employment / 365.25
    a["credit_income_ratio"] = ratio(a.AMT_CREDIT, a.AMT_INCOME_TOTAL)
    a["annuity_income_ratio"] = ratio(a.AMT_ANNUITY, a.AMT_INCOME_TOTAL)
    # Negative offsets denote events before the current application. Current snapshots
    # have unresolved as-of semantics; a real bank needs timestamped snapshots.
    b = frames["bureau"].loc[lambda d: d.DAYS_CREDIT < 0].copy()
    for c in ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE"]:
        b[c] = b[c].where(b[c] >= 0)
    b = b.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "size"),
        bureau_credit_sum=("AMT_CREDIT_SUM", lambda s: s.sum(min_count=1)),
        bureau_debt_sum=("AMT_CREDIT_SUM_DEBT", lambda s: s.sum(min_count=1)),
        bureau_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
        bureau_days_mean=("DAYS_CREDIT", "mean"),
    )
    h = frames["previous_application"].loc[lambda d: d.DAYS_DECISION < 0].copy()
    h["refused"] = h.NAME_CONTRACT_STATUS.eq("Refused").astype(float)
    h["AMT_APPLICATION"] = h.AMT_APPLICATION.where(h.AMT_APPLICATION >= 0)
    h = h.groupby("SK_ID_CURR").agg(
        previous_count=("SK_ID_PREV", "size"),
        previous_amount_mean=("AMT_APPLICATION", "mean"),
        previous_refusal_rate=("refused", "mean"),
    )
    result = a.merge(b, on="SK_ID_CURR", how="left", validate="one_to_one").merge(
        h, on="SK_ID_CURR", how="left", validate="one_to_one"
    )
    result["no_bureau_history"] = result.bureau_count.isna().astype(int)
    result["no_previous_history"] = result.previous_count.isna().astype(int)
    for c in ["bureau_count", "previous_count"]:
        result[c] = result[c].fillna(0)
    if len(result) != len(a) or set(result.SK_ID_CURR) != set(a.SK_ID_CURR):
        raise ValueError("Merge changed applicant population")
    assert "TARGET" not in FEATURES and "SK_ID_CURR" not in FEATURES
    result = result[["SK_ID_CURR", "TARGET"] + FEATURES].replace([np.inf, -np.inf], np.nan)
    return result
