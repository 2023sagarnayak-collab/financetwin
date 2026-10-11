import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models.health_score import engineer_features, compute_health_score
from engine.recommender import generate_recommendations

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

st.set_page_config(page_title="FinanceTwin", page_icon="📊", layout="wide")

# Same weights as models/health_score.py
WEIGHTS = {
    "Debt-to-income": 0.45,
    "EMI burden": 0.35,
    "Savings rate": 0.10,
    "Investment ratio": 0.05,
    "Credit utilization": 0.05,
}


# ---------- data loading ----------
@st.cache_data
def load_scored_data():
    credit = pd.read_csv(os.path.join(PROCESSED_DIR, "credit_clean.csv"))
    credit = engineer_features(credit)
    return compute_health_score(credit)


@st.cache_data
def load_rfm_clusters():
    rfm = pd.read_csv(os.path.join(PROCESSED_DIR, "customer_clustered.csv"))

    feat = pd.DataFrame({
        "Recency": rfm["Recency"],
        "Frequency": np.log1p(rfm["Frequency"]),
        "Monetary": np.log1p(rfm["Monetary"]),
    })
    means, stds = feat.mean(), feat.std()
    scaled = (feat - means) / stds

    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    rfm["cluster"] = model.fit_predict(scaled)

    # Label by behaviour, not by cluster number
    active_id = rfm.groupby("cluster")["Monetary"].mean().idxmax()
    rfm["segment"] = np.where(rfm["cluster"] == active_id, "Active / Engaged", "Lapsed / One-time")
    return rfm, model, means, stds, active_id


# ---------- helpers ----------
def normalize_against(value, reference, invert=False):
    lo, hi = reference.quantile(0.01), reference.quantile(0.99)
    scaled = float(np.clip((value - lo) / (hi - lo), 0, 1))
    return 1 - scaled if invert else scaled


def score_user(income, salary, debt, balance, invested, emi, credit_util, data):
    metrics = {
        "debt_to_income": debt / income,
        "emi_burden": emi / salary,
        "savings_rate": balance / salary,
        "investment_ratio": invested / salary,
        "credit_utilization": credit_util / 100,
    }
    sub = {
        "Debt-to-income": normalize_against(metrics["debt_to_income"], data["debt_to_income"], invert=True),
        "EMI burden": normalize_against(metrics["emi_burden"], data["emi_burden"], invert=True),
        "Savings rate": normalize_against(metrics["savings_rate"], data["savings_rate"]),
        "Investment ratio": normalize_against(metrics["investment_ratio"], data["investment_ratio"]),
        "Credit utilization": normalize_against(metrics["credit_utilization"], data["credit_utilization"], invert=True),
    }
    score = sum(sub[k] * WEIGHTS[k] for k in sub) * 100
    return metrics, sub, score


# ---------- pages ----------
def page_overview():
    st.title("FinanceTwin")
    st.subheader("An intelligent personal financial planning and decision support system")
    st.write(
        "Most budgeting apps only record what you already spent. FinanceTwin looks forward: "
        "it scores your financial health, forecasts spending, groups spending behaviour, "
        "and suggests what to improve."
    )

    data = load_scored_data()
    m1, m2, m3 = st.columns(3)
    m1.metric("Users scored", f"{len(data):,}")
    m2.metric("Average Health Score", f"{data['health_score'].mean():.1f}")
    m3.metric("Median Health Score", f"{data['health_score'].median():.1f}")

    st.subheader("How Health Scores are spread across all users")
    counts, edges = np.histogram(data["health_score"], bins=25, range=(0, 100))
    hist = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "Users": counts})
    chart = alt.Chart(hist).mark_bar().encode(
        x=alt.X("start:Q", title="Health Score", scale=alt.Scale(domain=[0, 100])),
        x2="end:Q",
        y=alt.Y("Users:Q", title="Number of users"),
        tooltip=["start", "end", "Users"],
    )
    st.altair_chart(chart)

    st.subheader("What you can do here")
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**My Financial Health**")
        st.write("Enter your numbers, get your score, and see personalised suggestions.")
    with b.container(border=True):
        st.markdown("**Spending Forecast**")
        st.write("See past monthly spending and a projection of the next months.")
    c, d = st.columns(2)
    with c.container(border=True):
        st.markdown("**Customer Segments**")
        st.write("See how customers split into active and lapsed spending groups.")
    with d.container(border=True):
        st.markdown("**About & Methodology**")
        st.write("Datasets, validation results and honest limitations.")


def page_health():
    st.title("My Financial Health")
    st.write("Enter your monthly figures in rupees. Your score and suggestions update as you change a value.")
    st.caption(
        "The score is built from ratios (for example debt divided by income), "
        "so it does not depend on the currency you use."
    )
    data = load_scored_data()

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.subheader("Your numbers")
        st.markdown("**Income**")
        income = st.slider("Annual income (₹)", 100000, 3000000, 600000, step=10000)
        salary = st.slider("Monthly in-hand salary (₹)", 5000, 250000, 40000, step=1000)
        st.markdown("**Debt**")
        debt = st.slider("Outstanding debt (₹)", 0, 2000000, 50000, step=10000)
        emi = st.slider("Total EMI per month (₹)", 0, 150000, 8000, step=1000)
        credit_util = st.slider("Credit utilization (%)", 0, 100, 25)
        st.markdown("**Savings and investing**")
        balance = st.slider("Monthly balance left over (₹)", 0, 100000, 15000, step=1000)
        invested = st.slider("Invested per month (₹)", 0, 50000, 5000, step=500)

    metrics, sub, score = score_user(income, salary, debt, balance, invested, emi, credit_util, data)
    pct = float((data["health_score"] < score).mean() * 100)

    with right:
        st.subheader("Your results")
        r1, r2 = st.columns(2)
        r1.metric("Health Score", f"{score:.0f} / 100")
        r2.metric("Better than", f"{pct:.0f}% of users")
        st.progress(min(max(score, 0), 100) / 100)

        if pct >= 75:
            st.success("Strong: your finances look healthier than most users in the dataset.")
        elif pct >= 25:
            st.info("Moderate: you are around the middle of the pack.")
        else:
            st.warning("Needs attention: your score is in the lowest quarter of users.")

        st.markdown("**How you score on each factor** (higher is better)")
        breakdown = pd.DataFrame({
            "Factor": list(sub),
            "Score": [round(v * 100, 1) for v in sub.values()],
            "Weight": [f"{WEIGHTS[k]:.0%}" for k in sub],
        })
        bars = alt.Chart(breakdown).mark_bar().encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Score (0-100)"),
            y=alt.Y("Factor:N", sort="-x", title=None),
            tooltip=["Factor", "Score", "Weight"],
        )
        st.altair_chart(bars)
        st.caption(
            "Debt-to-income and EMI burden carry 80% of the total weight because they were the "
            "factors that best separated good from poor credit in the data."
        )

        st.markdown("**Personalised suggestions**")
        for rec in generate_recommendations(metrics):
            text = f"**{rec['title']}**\n\n{rec['message']}"
            priority = rec.get("priority")
            if priority is None:
                st.success(text)
            elif priority <= 2:
                st.warning(text)
            else:
                st.info(text)


def page_forecast():
    st.title("Spending Forecast")
    try:
        from models.forecasting import load_monthly_series, forecast_future
    except ImportError as e:
        st.error(
            f"Forecasting needs the 'statsmodels' package ({e}). Start the app from your "
            "virtual environment with: python -m streamlit run dashboard/app.py"
        )
        return

    st.write("Monthly spending from the personal budget dataset, with a simple projection.")
    st.caption("Amounts are in the dataset's own units; its currency is not specified.")
    periods = st.slider("Months to forecast", 1, 6, 3)

    monthly = load_monthly_series()
    future = forecast_future(monthly, periods=periods)

    def naive(index):
        return pd.DatetimeIndex(index).tz_localize(None)

    actual = pd.DataFrame({"Month": naive(monthly.index), "Spend": monthly.values, "Type": "Actual"})
    bridge = pd.DataFrame({"Month": [actual["Month"].iloc[-1]], "Spend": [monthly.iloc[-1]], "Type": "Forecast"})
    fc = pd.DataFrame({"Month": naive(future.index), "Spend": future.values, "Type": "Forecast"})
    plot_df = pd.concat([actual, bridge, fc], ignore_index=True)

    line = alt.Chart(plot_df).mark_line(point=True).encode(
        x=alt.X("Month:T", title=None),
        y=alt.Y("Spend:Q", title="Monthly spend (dataset units)"),
        color=alt.Color("Type:N", title=None),
        tooltip=["Month:T", "Spend:Q", "Type:N"],
    )
    st.altair_chart(line)

    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast per month", f"{future.iloc[0]:,.0f}")
    m2.metric("Average of last 6 months", f"{monthly.iloc[-6:].mean():,.0f}")
    m3.metric("Highest month on record", f"{monthly.max():,.0f}")

    st.info(
        "The forecast is flat on purpose. Backtesting showed this spending has no reliable "
        "trend, so the best estimate is a smoothed recent level."
    )

    with st.expander("How accurate is this forecast?"):
        st.write(
            "Each method was trained on all but the last 6 months and tested on those 6 months. "
            "Lower error is better."
        )
        backtest = pd.DataFrame({
            "Method": ["Naive (repeat last month)", "Smoothing, no trend (used here)",
                       "Smoothing with trend", "3-month moving average"],
            "Error (RMSE)": [674.57, 688.04, 1260.55, 1518.17],
        })
        st.table(backtest.set_index("Method"))
        st.write(
            "No method beat the naive baseline. Individual monthly spending is noisy, so a "
            "forecast here is indicative only."
        )


def page_segments():
    st.title("Customer Spending Segments")
    st.write(
        "Customers are grouped by three behaviours (RFM): **Recency** (days since last purchase), "
        "**Frequency** (number of orders) and **Monetary** (total spend). This uses a separate "
        "retail dataset with real repeat purchases."
    )
    st.caption("Spend amounts are in the retail dataset's own units, so they are not rupees.")
    rfm, model, means, stds, active_id = load_rfm_clusters()

    counts = rfm["segment"].value_counts()
    c1, c2 = st.columns(2)
    c1.metric("Active / Engaged customers", f"{counts.get('Active / Engaged', 0):,}")
    c2.metric("Lapsed / One-time customers", f"{counts.get('Lapsed / One-time', 0):,}")

    summary = rfm.groupby("segment").agg(**{
        "Avg days since last purchase": ("Recency", "mean"),
        "Avg number of orders": ("Frequency", "mean"),
        "Avg total spend (dataset units)": ("Monetary", "mean"),
    }).round(1)
    st.table(summary)

    st.subheader("Where customers sit")
    sample = rfm.sample(min(len(rfm), 3000), random_state=42)[["Recency", "Frequency", "Monetary", "segment"]]
    scatter = alt.Chart(sample).mark_circle(size=40, opacity=0.5).encode(
        x=alt.X("Recency:Q", title="Days since last purchase"),
        y=alt.Y("Monetary:Q", scale=alt.Scale(type="log"), title="Total spend (dataset units, log scale)"),
        color=alt.Color("segment:N", title="Segment"),
        tooltip=["Recency", "Frequency", "Monetary", "segment"],
    )
    st.altair_chart(scatter)
    st.caption("Random sample of 3,000 customers. The spend axis is logarithmic because a few customers spend far more than the rest.")

    st.subheader("Classify a customer")
    a, b, c = st.columns(3)
    with a:
        recency = st.slider("Days since last purchase", 1, 740, 90)
    with b:
        frequency = st.number_input("Number of orders", min_value=1, max_value=400, value=5)
    with c:
        monetary = st.number_input("Total spend (dataset units)", min_value=1, max_value=1000000, value=1000, step=100)

    row = pd.DataFrame([{
        "Recency": recency,
        "Frequency": np.log1p(frequency),
        "Monetary": np.log1p(monetary),
    }])
    cluster = model.predict((row - means) / stds)[0]
    if cluster == active_id:
        st.success("Predicted segment: **Active / Engaged**. Recent, frequent and higher-spending behaviour.")
    else:
        st.warning("Predicted segment: **Lapsed / One-time**. Long gap since last purchase, few orders, lower spend.")


def page_about():
    st.title("About & Methodology")
    st.markdown(
        """
**FinanceTwin** combines finance concepts with data science to move beyond expense tracking.

### Modules and datasets
| Module | Data used | Method |
|---|---|---|
| Financial Health Score | Credit score classification dataset (Kaggle) | Weighted score from debt, EMI, savings, investment and credit-use ratios |
| Spending forecast | Personal budget transactions (Kaggle) | Exponential smoothing, backtested against baselines |
| Customer segments | Online Retail II (Kaggle) | RFM features, log-scaling, k-means |
| Recommendations | Output of the Health Score | Transparent rule-based thresholds |

### Key findings
- **Health Score validation:** average scores follow the real credit labels (Good about 82, Standard about 79, Poor about 75). Debt-to-income and EMI burden were the factors that separated these groups; savings, investment and credit utilization were weak in this dataset.
- **Forecasting:** no method beat the naive "repeat last month" baseline. Individual monthly spending is noisy with one-off spikes.
- **Segmentation:** a first dataset with one transaction per customer showed no link between age and spend, so it was replaced by retail data with repeat purchases.

### Limitations
- Datasets are public and partly synthetic, so results show the method works rather than describing real people.
- About 18% of credit records were dropped because of missing salary values.
- Weights and recommendation thresholds are design choices and can be tuned. They are calibrated to the benchmark dataset and would need recalibration for region-specific borrowing patterns.
- Amounts on the Forecast and Segments pages are in the source datasets' own units (currency not standardised). The Health Score uses ratios, so it is unit-independent.
- This is a decision-support tool, not financial advice.
"""
    )


# ---------- navigation ----------
PAGES = {
    "Overview": page_overview,
    "My Financial Health": page_health,
    "Spending Forecast": page_forecast,
    "Customer Segments": page_segments,
    "About & Methodology": page_about,
}

st.sidebar.title("FinanceTwin")
choice = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("B.Tech Data Science · Interdisciplinary Project")

PAGES[choice]()