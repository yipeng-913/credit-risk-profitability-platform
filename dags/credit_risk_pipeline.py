"""Local sequential execution; run-specific Airflow logs, mode-specific local artifacts."""

from datetime import timedelta
import os
from pathlib import Path
import sys
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import stage

with DAG(
    "credit_risk_pipeline",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "credit-risk-lab", "retries": 1, "retry_delay": timedelta(seconds=15)},
    tags=["research", "synthetic-demo"],
) as dag:
    tasks = []
    for task_id, command in [
        ("validate_raw_data", "validate"),
        ("build_features", "build"),
        ("train_model", "train"),
        ("evaluate_model", "evaluate"),
        ("publish_local_artifacts", "publish"),
    ]:
        tasks.append(
            PythonOperator(
                task_id=task_id,
                python_callable=stage,
                op_kwargs={"name": command, "mode": os.getenv("CREDIT_MODE", "demo")},
            )
        )
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
