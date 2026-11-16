# EDA status

The published EDA is **synthetic demonstration only**: [generated findings](demo/eda_findings.md), [quality counts](demo/quality.json), [segment query output](demo/segments.csv), and six charts in this directory. Run real mode to produce actual local Home Credit findings under `artifacts/real/`; no real-data findings are currently claimed.

The implemented checks address employment sentinel 365243, missing external scores, invalid income/ranges, categorical imbalance and primary-key duplicates. Invalid continuous values become missing; numeric medians and categorical modes are learned within training folds. Unknown categories are safely encoded. History absence has explicit flags and zero counts; unavailable balances retain missingness. Duplicate primary keys are rejected rather than silently discarded. Source columns are allowlisted, and neither TARGET nor pseudonymous IDs enter the model.

The generated report summarizes five associations (credit/income, external score, age, bureau history and previous refusals). A correlation in synthetic data validates display behavior, not a lending conclusion.
