# 简历可用描述

**信贷风险与收益情景分析平台｜独立项目**  
基于 Python 搭建可复现的信贷风险分析原型，设计 Home Credit 申请、征信及历史申请三表接入流程，构建 15 项衍生特征并实现数据质量与关联基数校验；使用 Logistic Regression、LightGBM、分层交叉验证及独立校准/测试集完成建模评估，开发 Streamlit 三视图仪表盘与可调风险、LGD、资金成本的收益情景模拟器，并通过 Apache Airflow、Docker Compose 和 GitHub Actions 实现工作流编排与自动化验证，使用合成数据跑通端到端流程并通过 18 项测试。

技术栈：Python、Pandas、scikit-learn、LightGBM、DuckDB、Parquet、Streamlit、Airflow、Docker、Pytest、GitHub Actions。

真实性边界：当前为合成数据验证的独立项目，不代表已在 Kaggle 全量数据上验证，不属于 ING 实习或银行生产落地。真实数据接入并跑出结果后，再将实际样本量、指标补进简历；不要把演示 ROC-AUC 写成真实业务效果。

## English version

Built an independent, reproducible credit-risk and illustrative profitability prototype in Python, implementing applicant-level joins across application, bureau and prior-application tables with 15 engineered features and data-quality checks. Compared Logistic Regression and LightGBM using stratified cross-validation, separate probability calibration and held-out evaluation; developed a three-view Streamlit dashboard and configurable financial scenario engine; orchestrated the workflow with Airflow and Docker Compose and automated verification with GitHub Actions, demonstrating the end-to-end system on synthetic data with 18 passing tests.
