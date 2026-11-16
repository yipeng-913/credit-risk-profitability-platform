# EDA findings — SYNTHETIC DEMONSTRATION

Applicants: 1800; unique IDs: 1800; event rate: 11.61%.

TARGET represents dataset-specific repayment difficulty. Accuracy hides minority-event errors.

Quality checks cover employment sentinel, missing external scores, invalid income, invalid score range, categorical imbalance, and duplicate IDs (rejected upstream). See quality.json for actual counts.

Associations below are descriptive, not causal. Bins use the local sample, and small groups are unstable.

- credit_income_ratio: observed event rates by ascending quantile: 4.89%, 7.11%, 14.44%, 20.00%.

- EXT_SOURCE_2: observed event rates by ascending quantile: 15.01%, 13.11%, 10.44%, 8.23%.

- age_years: observed event rates by ascending quantile: 12.89%, 10.44%, 12.44%, 10.67%.

- bureau_count: observed event rates by ascending quantile: 11.65%, 8.53%, 14.95%, 12.08%.

- previous_refusal_rate: observed event rates by ascending quantile: 11.46%, 10.89%, 13.54%.

Stakeholder relevance: investigate affordability ratios; review external-score missingness; validate historical-credit signals using timestamped snapshots.

Synthetic findings validate software behavior only. No conclusion about real borrowers follows.
