-- Descriptive association on locally labeled data; no causal or decision claim.
SELECT NAME_INCOME_TYPE AS income_segment,
       COUNT(*) AS applicants, AVG(TARGET) AS observed_difficulty_rate,
       AVG(credit_income_ratio) AS mean_credit_income_ratio,
       AVG(CASE WHEN EXT_SOURCE_2 IS NULL THEN 1.0 ELSE 0.0 END) AS external_score_missing_rate
FROM read_parquet(?)
GROUP BY NAME_INCOME_TYPE ORDER BY applicants DESC;
