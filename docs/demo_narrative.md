# Six-part technical demo · approximately 10 minutes

## 1. Problem and scope · 1 minute
An independent research project, not ING work. TARGET is repayment difficulty, and all current evidence is synthetic. The product question is how offline risk evidence and financial assumptions interact.

## 2. Data and feature integrity · 2 minutes
Show validation and the two history tables. Explain why direct one-to-many joins inflate applicant rows. Demonstrate aggregation, one-to-one joins, 15 engineered features, missingness policy and source allowlisting. A negative date offset is useful but insufficient proof of as-of availability.

## 3. Evaluation design · 2 minutes
Show 60/20/20 IDs and training-only cross-validation. Logistic Regression beats LightGBM in this synthetic experiment. Explain sigmoid calibration, AP versus ROC-AUC, final hold-out isolation, and why application_test has no usable labels. Show the confusion matrix and discuss false approvals versus missed suitable customers without claiming an operational threshold.

## 4. Stakeholder experience · 2 minutes
Walk through all three Streamlit views. Applicant medians and global importance provide context, not causal local attribution. Move risk, funding and LGD assumptions and show the financial components. Explain why a dataset score is not regulatory PD.

## 5. Reproducibility · 2 minutes
Show passing CI, the Airflow report and Docker instructions. Explain retries, all-success dependencies, deterministic overwrites, publication readiness and artifact hashes. A successful train followed by failed evaluation must not publish a new usable bundle.

## 6. Decision and next steps · 1 minute
No real-loan decisions. Obtain authorized timestamped data, validate outcomes and segments, collect actual economics and require independent validation. With two more weeks, prioritize a real-data run and temporal validation over additional models. The finance assumptions are the least validated business component.
