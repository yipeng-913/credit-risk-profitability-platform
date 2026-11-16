import json
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import duckdb

ROOT = Path(__file__).resolve().parents[2]


def report(features, raw, output, mode):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(features)
    app = pd.read_csv(Path(raw) / "application_train.csv")
    title = "SYNTHETIC DEMONSTRATION" if mode == "demo" else "HOME CREDIT LOCAL SAMPLE"
    charts = {
        "01_target": ("TARGET", "count"),
        "02_income": ("AMT_INCOME_TOTAL", "hist"),
        "03_credit_income": ("credit_income_ratio", "hist"),
        "04_external_score": ("EXT_SOURCE_2", "risk"),
        "05_age": ("age_years", "risk"),
        "06_history": ("bureau_count", "risk"),
    }
    for name, (col, kind) in charts.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        if kind == "count":
            df[col].value_counts().sort_index().plot.bar(ax=ax)
        elif kind == "hist":
            df[col].clip(upper=df[col].quantile(0.99)).hist(ax=ax, bins=30)
        else:
            temp = df.assign(bin=pd.qcut(df[col], 5, duplicates="drop"))
            temp.groupby("bin", observed=True).TARGET.mean().plot.bar(ax=ax)
            ax.set_ylabel("Observed difficulty rate")
        ax.set_title(f"{title}: {col}")
        fig.tight_layout()
        fig.savefig(output / f"{name}.png")
        plt.close(fig)
    quality = dict(
        applicants=len(df),
        unique_ids=int(df.SK_ID_CURR.nunique()),
        event_rate=float(df.TARGET.mean()),
        employment_sentinel=int(app.DAYS_EMPLOYED.eq(365243).sum()),
        nonpositive_income=int(app.AMT_INCOME_TOTAL.le(0).sum()),
        invalid_external_score=int((app.EXT_SOURCE_2.notna() & ~app.EXT_SOURCE_2.between(0, 1)).sum()),
        missing_fraction=df.isna().mean().sort_values(ascending=False).head(10).to_dict(),
        income_type_counts=df.NAME_INCOME_TYPE.value_counts().to_dict(),
    )
    (output / "quality.json").write_text(json.dumps(quality, indent=2))
    con = duckdb.connect()
    seg = con.execute((ROOT / "sql/portfolio_segments.sql").read_text(), [str(features)]).df()
    seg.to_csv(output / "segments.csv", index=False)
    lines = [
        f"# EDA findings — {title}",
        f"Applicants: {len(df)}; unique IDs: {df.SK_ID_CURR.nunique()}; event rate: {df.TARGET.mean():.2%}.",
        "TARGET represents dataset-specific repayment difficulty. Accuracy hides minority-event errors.",
        "Quality checks cover employment sentinel, missing external scores, invalid income, invalid score range, categorical imbalance, and duplicate IDs (rejected upstream). See quality.json for actual counts.",
        "Associations below are descriptive, not causal. Bins use the local sample, and small groups are unstable.",
    ]
    for col in ["credit_income_ratio", "EXT_SOURCE_2", "age_years", "bureau_count", "previous_refusal_rate"]:
        temp = df.assign(bin=pd.qcut(df[col], 4, duplicates="drop"))
        rates = temp.groupby("bin", observed=True).TARGET.mean()
        lines.append(
            f"- {col}: observed event rates by ascending quantile: {', '.join(f'{x:.2%}' for x in rates)}."
        )
    lines += [
        "Stakeholder relevance: investigate affordability ratios; review external-score missingness; validate historical-credit signals using timestamped snapshots.",
        "Synthetic findings validate software behavior only. No conclusion about real borrowers follows.",
    ]
    (output / "eda_findings.md").write_text("\n\n".join(lines) + "\n")
