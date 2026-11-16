from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.inspection import permutation_importance
from src.features.build import FEATURES

SEED = 42
THRESHOLD = 0.20  # Demonstration threshold fixed before hold-out; not an approval rule.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimator(kind):
    numeric = [c for c in FEATURES if c != "NAME_INCOME_TYPE"]
    pre = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "category",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["NAME_INCOME_TYPE"],
            ),
        ]
    )
    model = (
        LogisticRegression(max_iter=1500, random_state=SEED)
        if kind == "logistic"
        else LGBMClassifier(
            n_estimators=100,
            num_leaves=15,
            max_depth=5,
            min_child_samples=25,
            learning_rate=0.04,
            random_state=SEED,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def train(feature_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(feature_path).sort_values("SK_ID_CURR").reset_index(drop=True)
    train_cal, hold = train_test_split(df.index, test_size=0.20, stratify=df.TARGET, random_state=SEED)
    train_idx, cal = train_test_split(
        train_cal, test_size=0.25, stratify=df.loc[train_cal, "TARGET"], random_state=SEED
    )
    splits = {
        k: df.loc[v, "SK_ID_CURR"].astype(int).tolist()
        for k, v in [("train", train_idx), ("calibration", cal), ("holdout", hold)]
    }
    (output / "splits.json").write_text(json.dumps(splits, indent=2))
    X, y = df.loc[train_idx, FEATURES], df.loc[train_idx, "TARGET"]
    candidates = {}
    fitted = {}
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    for kind in ["logistic", "lightgbm"]:
        pipe = estimator(kind)
        scores = cross_validate(
            pipe, X, y, cv=cv, scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"}, n_jobs=1
        )
        candidates[kind] = {k: float(np.mean(scores["test_" + k])) for k in ["roc_auc", "pr_auc"]}
        fitted[kind] = pipe.fit(X, y)
    # Selection uses training-only CV. Calibration and final hold-out never select a model.
    winner = max(candidates, key=lambda k: candidates[k]["pr_auc"])
    model = CalibratedClassifierCV(FrozenEstimator(fitted[winner]), method="sigmoid")
    model.fit(df.loc[cal, FEATURES], df.loc[cal, "TARGET"])
    importance = permutation_importance(
        model,
        df.loc[cal, FEATURES],
        df.loc[cal, "TARGET"],
        scoring="roc_auc",
        n_repeats=3,
        random_state=SEED,
        n_jobs=1,
    )
    pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_auc_drop": importance.importances_mean,
            "std_auc_drop": importance.importances_std,
        }
    ).sort_values("mean_auc_drop", ascending=False).to_csv(output / "importance.csv", index=False)
    joblib.dump(model, output / "model.joblib")
    manifest = {
        "seed": SEED,
        "threshold": THRESHOLD,
        "selection": "training-only 3-fold CV average precision",
        "selected_model": winner,
        "cv": candidates,
        "feature_sha256": digest(feature_path),
        "model_sha256": digest(output / "model.joblib"),
        "splits_sha256": digest(output / "splits.json"),
        "features": FEATURES,
        "split_counts": {k: len(v) for k, v in splits.items()},
    }
    (output / "training.json").write_text(json.dumps(manifest, indent=2))
    return manifest
