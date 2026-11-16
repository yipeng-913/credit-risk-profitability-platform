# Verification report

Verified on 2026-11-16 using Python 3.12.14 on Linux. All model evidence uses synthetic data.

- Ruff: passed.
- Pytest: 18 passed; [JUnit report](test-results.xml). Tests include deliberate invalid data and financial boundary cases.
- Airflow 2.10.5: a real local `airflow dags test` run executed all five PythonOperator tasks successfully. [Task states and timestamps queried from the Airflow metadata database](airflow_run.json).
- Playwright with Chromium: all three dashboard views rendered without Streamlit exceptions; changed the loan amount and selected another applicant; selecting real mode without artifacts showed the intended error. [Screenshots](screenshots/).
- Synthetic pipeline: 1,800 rows, reloadable model, offline final hold-out metrics, six EDA charts, calibration, confusion matrix, DuckDB segment query, financial scenarios and readiness manifest generated successfully.

Airflow local task evidence:

| Task | Status |
|---|---|
| `validate_raw_data` | success |
| `build_features` | success |
| `train_model` | success |
| `evaluate_model` | success |
| `publish_local_artifacts` | success |

`dags test` is actual local DAG execution, not proof of a long-running production scheduler. CI repeats this command. GitHub Actions status is available from the live repository badge. Container validation is recorded below after the build check.
