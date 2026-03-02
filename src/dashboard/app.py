"""Streamlit dashboard for credit risk scoring visualization.

Displays credit score distributions, model performance, scorecard
points, fairness metrics, and WoE analysis using synthetic demo data.

Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RISK_BANDS = ["Excellent", "Good", "Fair", "Poor", "Very Poor"]
RISK_COLORS = ["#4CAF50", "#8BC34A", "#FF9800", "#FF5722", "#F44336"]

SCORECARD_FEATURES = [
    "Age",
    "Income",
    "Employment Length",
    "Loan Amount",
    "Interest Rate",
    "DTI Ratio",
    "Delinquencies",
    "Credit History",
]


def generate_score_distribution(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic credit score distribution."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(650, 80, size=1000)
    scores = np.clip(scores, 300, 850).astype(int)
    return pd.DataFrame({"score": scores})


def generate_model_metrics(seed: int = 42) -> dict:
    """Generate synthetic model performance metrics."""
    rng = np.random.default_rng(seed)
    auc = round(rng.uniform(0.78, 0.88), 4)
    return {
        "auc_roc": auc,
        "gini": round(2 * auc - 1, 4),
        "ks_statistic": round(rng.uniform(0.35, 0.55), 4),
        "accuracy": round(rng.uniform(0.78, 0.86), 4),
        "precision": round(rng.uniform(0.72, 0.82), 4),
        "recall": round(rng.uniform(0.65, 0.78), 4),
    }


def generate_scorecard_points(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic scorecard point allocations."""
    rng = np.random.default_rng(seed)
    rows = []
    for feat in SCORECARD_FEATURES:
        points = int(rng.integers(-40, 60))
        iv = round(rng.uniform(0.02, 0.5), 4)
        rows.append({"feature": feat, "points": points, "iv": iv})
    return pd.DataFrame(rows)


def generate_fairness_metrics(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic fairness analysis metrics."""
    rng = np.random.default_rng(seed)
    groups = ["Age < 30", "Age 30-50", "Age > 50", "Male", "Female"]
    rows = []
    for group in groups:
        rows.append(
            {
                "group": group,
                "approval_rate": round(rng.uniform(0.55, 0.85), 4),
                "avg_score": int(rng.integers(580, 720)),
                "default_rate": round(rng.uniform(0.03, 0.15), 4),
                "demographic_parity_diff": round(rng.uniform(-0.1, 0.1), 4),
            }
        )
    return pd.DataFrame(rows)


def generate_pd_by_score(seed: int = 42) -> pd.DataFrame:
    """Generate probability of default by score range."""
    rng = np.random.default_rng(seed)
    ranges = ["300-400", "400-500", "500-600", "600-700", "700-800", "800-850"]
    pds = [0.35, 0.20, 0.12, 0.06, 0.02, 0.005]
    rows = []
    for r, pd_val in zip(ranges, pds, strict=True):
        rows.append(
            {
                "score_range": r,
                "probability_of_default": round(max(pd_val + rng.uniform(-0.01, 0.01), 0.001), 4),
                "count": int(rng.integers(50, 300)),
            }
        )
    return pd.DataFrame(rows)


def render_header() -> None:
    """Render the dashboard header."""
    st.title("Credit Risk Scoring Dashboard")
    st.caption(
        "WoE-based scorecard with model performance, fairness analysis, "
        "and regulatory compliance monitoring"
    )


def render_summary_metrics(metrics: dict, scores_df: pd.DataFrame) -> None:
    """Render top-level summary metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
    col2.metric("Gini Coefficient", f"{metrics['gini']:.4f}")
    col3.metric("KS Statistic", f"{metrics['ks_statistic']:.4f}")
    col4.metric("Avg Score", f"{scores_df['score'].mean():.0f}")


def render_score_distribution(scores_df: pd.DataFrame) -> None:
    """Render credit score distribution histogram."""
    st.subheader("Credit Score Distribution")
    fig = px.histogram(
        scores_df,
        x="score",
        nbins=40,
        color_discrete_sequence=["#2196F3"],
    )
    boundaries = [300, 500, 600, 700, 750, 850]
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=RISK_COLORS[len(RISK_BANDS) - 1 - i],
            opacity=0.1,
            line_width=0,
        )
    fig.update_layout(
        xaxis_title="Credit Score",
        yaxis_title="Count",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_scorecard_points(points_df: pd.DataFrame) -> None:
    """Render scorecard point allocation chart."""
    st.subheader("Scorecard Point Allocation")
    sorted_df = points_df.sort_values("points")
    fig = px.bar(
        sorted_df,
        x="points",
        y="feature",
        orientation="h",
        color="points",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
    )
    fig.update_layout(
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pd_curve(pd_df: pd.DataFrame) -> None:
    """Render probability of default by score range."""
    st.subheader("Default Probability by Score Range")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pd_df["score_range"],
            y=pd_df["probability_of_default"],
            marker_color=[
                "#F44336",
                "#FF5722",
                "#FF9800",
                "#FFC107",
                "#8BC34A",
                "#4CAF50",
            ],
            text=pd_df["probability_of_default"].apply(lambda x: f"{x:.1%}"),
            textposition="auto",
        )
    )
    fig.update_layout(
        yaxis={"tickformat": ".0%"},
        yaxis_title="Probability of Default",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_fairness_analysis(fairness_df: pd.DataFrame) -> None:
    """Render fairness metrics comparison."""
    st.subheader("Fairness Analysis")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Approval Rate",
            x=fairness_df["group"],
            y=fairness_df["approval_rate"],
            text=fairness_df["approval_rate"].apply(lambda x: f"{x:.1%}"),
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Default Rate",
            x=fairness_df["group"],
            y=fairness_df["default_rate"],
            text=fairness_df["default_rate"].apply(lambda x: f"{x:.1%}"),
            textposition="auto",
        )
    )
    fig.update_layout(
        barmode="group",
        yaxis={"tickformat": ".0%"},
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main dashboard entry point."""
    render_header()

    scores_df = generate_score_distribution()
    metrics = generate_model_metrics()
    points_df = generate_scorecard_points()
    fairness_df = generate_fairness_metrics()
    pd_df = generate_pd_by_score()

    render_summary_metrics(metrics, scores_df)
    st.markdown("---")

    render_score_distribution(scores_df)

    col_left, col_right = st.columns(2)
    with col_left:
        render_scorecard_points(points_df)
    with col_right:
        render_pd_curve(pd_df)

    st.markdown("---")
    render_fairness_analysis(fairness_df)


if __name__ == "__main__":
    main()
