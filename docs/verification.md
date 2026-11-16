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

`dags test` is actual local DAG execution, not proof of a long-running production scheduler. CI repeats this command. GitHub Actions status is available from the live repository badge. Container validation is recorded below.

## Container verification

The lightweight Python 3.12 image built successfully (image SHA `4c114afa61fc58637d5b77c60e1b7d44975d387704647e6ceec893c152b243c4`). `docker compose config --quiet` passed. This cloud environment requires its HTTPS inspection CA; the build used a temporary Dockerfile with a BuildKit secret mount and `PIP_CERT`, preserving TLS verification and keeping the CA out of image layers. The committed Dockerfile uses normal system trust for standard environments.

**Container runtime remains unverified here:** Docker uses a space-heavy VFS storage driver, and creating the writable container layer failed with `no space left on device` before any application command ran. Only task-created images and build cache were removed afterward. No successful Docker Compose service run is claimed. The Python pipeline and actual Airflow DAG execution succeeded directly in the local virtual environment.
