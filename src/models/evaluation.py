import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import calibration_curve
from src.features.build import FEATURES
from src.models.training import digest


def evaluate(feature_path, output):
    output = Path(output)
    meta = json.loads((output / "training.json").read_text())
    for field, path in [
        ("feature_sha256", feature_path),
        ("model_sha256", output / "model.joblib"),
        ("splits_sha256", output / "splits.json"),
    ]:
        if digest(path) != meta[field]:
            raise ValueError("Stale or modified artifact: " + field)
    df = pd.read_parquet(feature_path)
    split = json.loads((output / "splits.json").read_text())
    sets = [set(split[k]) for k in ["train", "calibration", "holdout"]]
    if any(sets[i] & sets[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError("Split overlap")
    if set.union(*sets) != set(df.SK_ID_CURR) or sum(map(len, sets)) != len(df):
        raise ValueError("Split coverage mismatch")
    test = df.set_index("SK_ID_CURR").loc[split["holdout"]].reset_index()
    # Fresh load in the evaluation process; only this stage reads hold-out labels for metrics.
    model = joblib.load(output / "model.joblib")
    prob = model.predict_proba(test[FEATURES])[:, 1]
    y = test.TARGET
    matrix = confusion_matrix(y, prob >= meta["threshold"], labels=[0, 1])
    metrics = dict(
        roc_auc=float(roc_auc_score(y, prob)),
        pr_auc=float(average_precision_score(y, prob)),
        brier=float(brier_score_loss(y, prob)),
        prevalence=float(y.mean()),
        threshold=meta["threshold"],
        confusion_matrix=matrix.tolist(),
        n_holdout=len(y),
        evaluation="Untouched local 20% hold-out; PR-AUC is average precision",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    test[["SK_ID_CURR", "TARGET"]].assign(risk_score=prob).to_parquet(
        output / "holdout_scores.parquet", index=False
    )
    frac, mean = calibration_curve(y, prob, n_bins=8, strategy="quantile")
    pd.DataFrame({"mean_score": mean, "observed_frequency": frac}).to_csv(
        output / "calibration.csv", index=False
    )
    fig, ax = plt.subplots()
    ax.plot(mean, frac, "o-", label="Local hold-out")
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.set(
        xlabel="Estimated repayment-difficulty score",
        ylabel="Observed event frequency",
        title="Offline hold-out calibration",
    )
    ax.legend()
    fig.savefig(output / "calibration.png", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y, prob, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y, prob, ax=axes[1])
    fig.savefig(output / "discrimination.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Predicted class",
        ylabel="Observed class",
        title=f"Demonstration threshold {meta['threshold']:.2f}",
    )
    fig.savefig(output / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    (output / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "feature_sha256": digest(feature_path),
                "model_sha256": digest(output / "model.joblib"),
                "metrics_sha256": digest(output / "metrics.json"),
                "scores_sha256": digest(output / "holdout_scores.parquet"),
            },
            indent=2,
        )
    )
    return metrics
