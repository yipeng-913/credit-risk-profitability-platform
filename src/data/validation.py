from pathlib import Path
import pandas as pd

SCHEMA = {
    "application_train": [
        "SK_ID_CURR",
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_2",
        "NAME_INCOME_TYPE",
    ],
    "bureau": [
        "SK_ID_CURR",
        "SK_ID_BUREAU",
        "DAYS_CREDIT",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "CREDIT_DAY_OVERDUE",
    ],
    "previous_application": [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "DAYS_DECISION",
        "AMT_APPLICATION",
        "NAME_CONTRACT_STATUS",
    ],
}


def validate_frames(frames):
    for name, cols in SCHEMA.items():
        df = frames[name]
        if not set(cols) <= set(df):
            raise ValueError(f"{name}: missing columns {set(cols) - set(df)}")
        key = {
            "application_train": "SK_ID_CURR",
            "bureau": "SK_ID_BUREAU",
            "previous_application": "SK_ID_PREV",
        }[name]
        if df[key].isna().any() or df[key].duplicated().any():
            raise ValueError(f"{name}: null or duplicate {key}")
        if df.SK_ID_CURR.isna().any():
            raise ValueError(f"{name}: null applicant key")
    y = frames["application_train"].TARGET
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("TARGET must contain both binary classes and no missing values")


def read_raw(root):
    root = Path(root)
    if not (root / "HomeCredit_columns_description.csv").is_file():
        raise ValueError("Missing official data dictionary (or synthetic demo dictionary)")
    frames = {name: pd.read_csv(root / f"{name}.csv", usecols=cols) for name, cols in SCHEMA.items()}
    validate_frames(frames)
    return frames
