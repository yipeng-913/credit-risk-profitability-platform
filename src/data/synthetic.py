"""Independently generated fixtures, unrelated to actual Home Credit records."""

from pathlib import Path
import numpy as np
import pandas as pd


def generate(root, n=1800, seed=42):
    if n < 100:
        raise ValueError("Use at least 100 synthetic applicants")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = np.arange(900000, 900000 + n)
    income = rng.lognormal(11.5, 0.45, n)
    credit = income * rng.uniform(0.5, 5, n)
    ext = rng.beta(3, 3, n)
    difficulty = 1 / (1 + np.exp(-(-3.1 + 0.38 * credit / income + 2 * (0.5 - ext))))
    app = pd.DataFrame(
        dict(
            SK_ID_CURR=ids,
            TARGET=rng.binomial(1, difficulty),
            AMT_INCOME_TOTAL=income,
            AMT_CREDIT=credit,
            AMT_ANNUITY=credit * 0.065,
            DAYS_BIRTH=-rng.integers(7300, 24000, n),
            DAYS_EMPLOYED=-rng.integers(1, 14000, n),
            EXT_SOURCE_2=ext,
            NAME_INCOME_TYPE=rng.choice(["Working", "Pensioner", "Commercial associate"], n),
        )
    )
    app.loc[rng.choice(n, n // 15, False), "DAYS_EMPLOYED"] = 365243
    app.loc[rng.choice(n, n // 12, False), "EXT_SOURCE_2"] = np.nan
    k = n * 3
    bureau = pd.DataFrame(
        dict(
            SK_ID_CURR=rng.choice(ids, k),
            SK_ID_BUREAU=np.arange(k),
            DAYS_CREDIT=-rng.integers(1, 3000, k),
            AMT_CREDIT_SUM=rng.lognormal(10, 1, k),
            AMT_CREDIT_SUM_DEBT=rng.lognormal(8, 1, k),
            CREDIT_DAY_OVERDUE=rng.poisson(0.4, k),
        )
    )
    prev = pd.DataFrame(
        dict(
            SK_ID_CURR=rng.choice(ids, k),
            SK_ID_PREV=np.arange(k),
            DAYS_DECISION=-rng.integers(1, 2000, k),
            AMT_APPLICATION=rng.lognormal(10, 0.8, k),
            NAME_CONTRACT_STATUS=rng.choice(["Approved", "Refused", "Canceled"], k, p=[0.65, 0.25, 0.1]),
        )
    )
    for name, df in [("application_train", app), ("bureau", bureau), ("previous_application", prev)]:
        df.to_csv(root / f"{name}.csv", index=False)
    pd.DataFrame(
        {
            "Row": [1],
            "Table": ["synthetic"],
            "Description": ["Generated demonstration only; not Kaggle data dictionary"],
        }
    ).to_csv(root / "HomeCredit_columns_description.csv", index=False)
    (root / "SYNTHETIC_DATA.txt").write_text(
        "All records independently generated. Not real Home Credit data.\n"
    )
