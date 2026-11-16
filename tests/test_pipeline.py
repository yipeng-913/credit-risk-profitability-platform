import json
import joblib
import numpy as np
import pytest
from src.features.build import build_features, FEATURES
from src.models.training import train
from src.models.evaluation import evaluate


def test_roundtrip_and_holdout_separation(frames, tmp_path):
    df = build_features(frames)
    feature = tmp_path / "features.parquet"
    df.to_parquet(feature)
    train(feature, tmp_path)
    split = json.loads((tmp_path / "splits.json").read_text())
    sets = [set(split[k]) for k in ["train", "calibration", "holdout"]]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert set.union(*sets) == set(df.SK_ID_CURR)
    model = joblib.load(tmp_path / "model.joblib")
    before = model.predict_proba(df[FEATURES])[:, 1]
    metrics = evaluate(feature, tmp_path)
    assert metrics["n_holdout"] == 60
    assert 0 <= metrics["roc_auc"] <= 1 and 0 <= metrics["brier"] <= 1
    assert sum(map(sum, metrics["confusion_matrix"])) == 60
    after = joblib.load(tmp_path / "model.joblib").predict_proba(df[FEATURES])[:, 1]
    assert np.allclose(before, after)
    df.loc[0, "AMT_CREDIT"] += 1
    df.to_parquet(feature)
    with pytest.raises(ValueError, match="modified"):
        evaluate(feature, tmp_path)


def test_airflow_dependencies():
    from airflow.models import DagBag

    bag = DagBag(dag_folder="dags", include_examples=False)
    assert not bag.import_errors
    dag = bag.dags["credit_risk_pipeline"]
    assert dag.task_dict["validate_raw_data"].downstream_task_ids == {"build_features"}
    assert dag.task_dict["evaluate_model"].upstream_task_ids == {"train_model"}
    assert dag.task_dict["publish_local_artifacts"].upstream_task_ids == {"evaluate_model"}
    assert len(dag.tasks) == 5
