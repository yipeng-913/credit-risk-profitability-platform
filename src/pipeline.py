"""Idempotent local stages shared by CLI and Airflow; artifacts isolated by mode."""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
from src.data.synthetic import generate
from src.data.validation import read_raw
from src.features.build import build_features
from src.models.training import train, digest
from src.models.evaluation import evaluate
from src.data.eda import report
from src.finance.engine import Assumptions, scenarios

ROOT = Path(__file__).resolve().parents[1]


def paths(mode):
    if mode not in ["demo", "real"]:
        raise ValueError("Mode must be demo or real")
    raw = ROOT / "data" / ("demo" if mode == "demo" else "raw")
    output = ROOT / "artifacts" / mode
    return raw, output


def stage(name, mode="demo"):
    raw, out = paths(mode)
    out.mkdir(parents=True, exist_ok=True)
    feature = out / "features.parquet"
    if name != "publish":
        (out / "READY.json").unlink(missing_ok=True)
    if name in ["build", "train", "evaluate", "publish"]:
        source = json.loads((out / "data_manifest.json").read_text())
        current = {p.name: digest(p) for p in sorted(raw.glob("*.csv"))}
        if current != source["raw_sha256"]:
            raise ValueError("Raw files changed; restart from validate")
    if name == "validate":
        if mode == "demo" and not raw.exists():
            generate(raw)
        if mode == "real" and (raw / "SYNTHETIC_DATA.txt").exists():
            raise ValueError("Synthetic fixtures cannot be used in real mode")
        frames = read_raw(raw)
        (out / "data_manifest.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "rows": {k: len(v) for k, v in frames.items()},
                    "raw_sha256": {p.name: digest(p) for p in sorted(raw.glob("*.csv"))},
                },
                indent=2,
            )
        )
    elif name == "build":
        build_features(read_raw(raw)).to_parquet(feature, index=False)
    elif name == "train":
        train(feature, out)
    elif name == "evaluate":
        evaluate(feature, out)
    elif name == "publish":
        # Publish checks provenance before exposing the local artifact set.
        meta = json.loads((out / "training.json").read_text())
        for field, p in [
            ("feature_sha256", feature),
            ("model_sha256", out / "model.joblib"),
            ("splits_sha256", out / "splits.json"),
        ]:
            if digest(p) != meta[field]:
                raise ValueError("Artifact mismatch at publication")
        evidence = json.loads((out / "evaluation_manifest.json").read_text())
        for field, path in [
            ("feature_sha256", feature),
            ("model_sha256", out / "model.joblib"),
            ("metrics_sha256", out / "metrics.json"),
            ("scores_sha256", out / "holdout_scores.parquet"),
        ]:
            if digest(path) != evidence[field]:
                raise ValueError("Evaluation provenance mismatch")
        report(feature, raw, out, mode)
        pd.DataFrame(
            scenarios(
                Assumptions(
                    risk_proxy=float(pd.read_parquet(out / "holdout_scores.parquet").risk_score.mean())
                )
            )
        ).T.to_csv(out / "financial_scenarios.csv")
        (out / "READY.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "warning": "Research/demo only; score is not regulatory PD",
                    "artifact_sha256": {
                        p.name: digest(p)
                        for p in [
                            feature,
                            out / "model.joblib",
                            out / "metrics.json",
                            out / "holdout_scores.parquet",
                        ]
                    },
                },
                indent=2,
            )
        )
    else:
        raise ValueError("Unknown stage " + name)
    print(f"{mode}: {name} completed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "validate", "build", "train", "evaluate", "publish"])
    parser.add_argument("--mode", choices=["demo", "real"], default=os.getenv("CREDIT_MODE", "demo"))
    args = parser.parse_args()
    if args.command == "run":
        for s in ["validate", "build", "train", "evaluate", "publish"]:
            stage(s, args.mode)
    else:
        stage(args.command, args.mode)


if __name__ == "__main__":
    main()
