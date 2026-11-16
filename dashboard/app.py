from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.features.build import FEATURES
from src.finance.engine import Assumptions, calculate, scenarios
from src.pipeline import paths

st.set_page_config(page_title="Credit Risk Lab", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.stApp {background:#f5f7fb} h1,h2,h3 {color:#11233f}
[data-testid="stMetric"] {background:white;padding:18px;border-radius:12px;border:1px solid #e0e6ee}
</style>""",
    unsafe_allow_html=True,
)
st.sidebar.title("◈ Credit Risk Lab")
st.sidebar.caption("RISK · EVIDENCE · SCENARIOS")
mode = st.sidebar.selectbox("Data mode", ["demo", "real"])
view = st.sidebar.radio(
    "Workspace", ["Portfolio Overview", "Applicant Risk Explorer", "Profitability Simulator"]
)
st.sidebar.divider()
st.sidebar.info("Independent research project. No ING affiliation. No automated lending decisions.")
raw, out = paths(mode)
st.title(view)
st.caption("CREDIT RISK & PROFITABILITY ANALYTICS PLATFORM")
if mode == "demo":
    st.warning(
        "SYNTHETIC DEMONSTRATION — independently generated records and illustrative EUR amounts. Metrics do not describe Home Credit performance."
    )
else:
    st.info(
        "Home Credit local analysis. TARGET is repayment difficulty; the score is not a validated regulatory one-year PD."
    )
if not (out / "READY.json").exists():
    st.error(f"Artifacts are missing. Run: python -m src.pipeline run --mode {mode}")
    st.stop()
try:
    df = pd.read_parquet(out / "features.parquet")
    scores = pd.read_parquet(out / "holdout_scores.parquet")
    metrics = json.loads((out / "metrics.json").read_text())
    meta = json.loads((out / "training.json").read_text())
except (OSError, ValueError) as exc:
    st.error(f"Artifacts cannot be read: {exc}. Rerun the pipeline.")
    st.stop()

if view == "Portfolio Overview":
    cols = st.columns(4)
    for c, label, value in zip(
        cols,
        ["Labeled applicants", "Observed event rate", "Hold-out ROC-AUC", "Hold-out Brier score"],
        [f"{len(df):,}", f"{df.TARGET.mean():.1%}", f"{metrics['roc_auc']:.3f}", f"{metrics['brier']:.3f}"],
    ):
        c.metric(label, value)
    st.caption(
        "Event rate: share of TARGET=1 in the labeled local sample. ROC-AUC: ranking quality. Brier: mean squared probability error, lower is better. Offline hold-out only."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Affordability distribution")
        fig = px.histogram(df, x="credit_income_ratio", nbins=35, color_discrete_sequence=["#156c8a"])
        fig.update_layout(
            xaxis_title="Credit / annual income (observed source units)", yaxis_title="Applicants"
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Offline probability calibration")
        cal = pd.read_csv(out / "calibration.csv")
        fig = px.line(cal, x="mean_score", y="observed_frequency", markers=True)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line={"dash": "dash", "color": "#aab5c4"},
            )
        )
        fig.update_layout(xaxis_title="Mean predicted score", yaxis_title="Observed difficulty frequency")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Score segments · local hold-out")
    seg = scores.assign(
        segment=pd.cut(
            scores.risk_score,
            [0, 0.1, 0.2, 1],
            include_lowest=True,
            labels=["Below 10%", "10–20%", "Above 20%"],
        )
    )
    st.dataframe(
        seg.groupby("segment", observed=False).agg(
            applicants=("TARGET", "size"),
            observed_event_rate=("TARGET", "mean"),
            mean_score=("risk_score", "mean"),
        ),
        use_container_width=True,
    )
    st.caption(
        "Score bins are descriptive demo bands, not approval policies. PR-AUC means average precision; the reference event prevalence is "
        + f"{metrics['prevalence']:.1%}."
    )
elif view == "Applicant Risk Explorer":
    st.subheader("Explore a pseudonymous application")
    key = st.selectbox("Local hold-out applicant ID", scores.SK_ID_CURR.tolist())
    row = df.loc[df.SK_ID_CURR.eq(key)]
    try:
        model = joblib.load(out / "model.joblib")
        score = float(model.predict_proba(row[FEATURES])[0, 1])
    except Exception as exc:
        st.error(f"Unable to score this application: {exc}")
        st.stop()
    a, b = st.columns([1, 2])
    with a:
        st.metric("Estimated repayment-difficulty score", f"{score:.1%}")
        st.caption(
            "Dataset-specific outcome estimate. This is not regulatory PD or a lending recommendation."
        )
    with b:
        compare = row[
            [
                "credit_income_ratio",
                "annuity_income_ratio",
                "EXT_SOURCE_2",
                "bureau_count",
                "previous_refusal_rate",
            ]
        ].T
        compare.columns = ["Selected applicant"]
        compare["Portfolio median"] = df[compare.index].median()
        st.dataframe(compare, use_container_width=True)
    st.subheader("What the model uses · global validation evidence")
    imp = pd.read_csv(out / "importance.csv").head(8).sort_values("mean_auc_drop")
    st.plotly_chart(
        px.bar(imp, x="mean_auc_drop", y="feature", orientation="h", color_discrete_sequence=["#156c8a"]),
        use_container_width=True,
    )
    st.caption(
        "Permutation importance: decrease in calibration-set ROC-AUC after shuffling a feature. It is global, not a causal or applicant-level explanation. Correlated features can share importance."
    )
    st.info(
        "Affordability ratios summarize the requested credit burden; external scores summarize a source-provided signal; history counts and refusal rates summarize prior applications. Comparisons show context, not reasons for a real credit decision."
    )
else:
    st.subheader("One-year illustrative loan economics")
    st.caption(
        "Synthetic EUR currency examples. Constant EAD; interest is amount × rate. No amortization, default timing, foregone interest, or tax model."
    )
    left, right = st.columns([1, 2])
    with left:
        amount = st.number_input(
            "Loan amount · synthetic EUR", min_value=1.0, max_value=10000000.0, value=10000.0, step=500.0
        )
        rate = st.slider("Annual interest rate", 0.0, 1.0, 0.12, 0.01)
        proxy = st.slider("Scenario risk proxy", 0.0, 1.0, float(scores.risk_score.mean()), 0.01)
        st.caption(
            "Initial proxy is the mean model score on the local hold-out; edit to explore assumptions. It is not regulatory PD."
        )
        lgd = st.slider("Loss severity / LGD assumption", 0.0, 1.0, 0.45, 0.01)
        funding = st.slider("Annual funding rate", 0.0, 1.0, 0.04, 0.01)
        operating = st.number_input("Annual operating cost · synthetic EUR", min_value=0.0, value=150.0)
    try:
        a = Assumptions(amount, rate, proxy, lgd, funding, operating)
        result = calculate(a)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    with right:
        st.metric("Illustrative annual profit · EUR", f"{result['profit']:,.0f}")
        components = pd.DataFrame(
            {
                "component": ["Interest income", "Funding", "Operations", "Expected loss proxy"],
                "synthetic EUR": [
                    result["interest_income"],
                    -result["funding_cost"],
                    -result["operating_cost"],
                    -result["expected_loss_proxy"],
                ],
            }
        )
        st.plotly_chart(
            px.bar(components, x="component", y="synthetic EUR", color="component"), use_container_width=True
        )
        st.subheader("Scenario comparison")
        st.dataframe(pd.DataFrame(scenarios(a)).T.round(2), use_container_width=True)
    st.info(
        "Expected loss proxy = risk proxy × LGD × EAD. A model score can inform a scenario only after explicit outcome and horizon caveats. Higher risk multiplies the proxy by 1.5; funding stress adds 2 percentage points; both are capped at 100%."
    )
with st.expander("Definitions, assumptions & limitations"):
    st.write(
        "PD is probability of a defined default over a defined horizon; LGD is loss as a fraction of exposure on default; EAD is exposure at default. This dataset label does not establish a regulatory one-year PD."
    )
    st.write(
        "Random stratified splits do not validate future stability. Source snapshot dates are ambiguous. Protected-group fairness, independent validation, privacy review, cost schedules and observed profit labels are missing. Do not auto-approve or reject real loans."
    )
