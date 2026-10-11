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
from engine.whatif import run_scenario, project_balance

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


def fmt_duration(months):
    if months is None:
        return "Not reachable"
    years, rem = divmod(int(months), 12)
    if years == 0:
        return f"{rem} mo"
    if rem == 0:
        return f"{years} yr"
    return f"{years} yr {rem} mo"


def fmt_money_delta(value):
    sign = "+" if value >= 0 else "-"
    return f"{sign}₹{abs(value):,.0f}"


# ---------- pages ----------
def page_overview():
    st.title("FinanceTwin")
    st.subheader("An intelligent personal financial planning and decision support system")
    st.write(
        "Most budgeting apps only record what you already spent. FinanceTwin looks forward: "
        "it scores your financial health, tests goals with what-if scenarios, forecasts "
        "spending, groups spending behaviour, and suggests what to improve."
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
    cards = [
        ("My Financial Health", "Enter your numbers, get your score, and see personalised suggestions."),
        ("What-If Simulator", "Set a savings goal and see how income, spending or saving changes affect the time to reach it."),
        ("Spending Forecast", "See past monthly spending and a projection of the next months."),
        ("Customer Segments", "See how customers split into active and lapsed spending groups."),
        ("About & Methodology", "Datasets, validation results and honest limitations."),
    ]
    slots = st.columns(3) + st.columns(3)
    for slot, (title, text) in zip(slots, cards):
        with slot.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)


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


def page_whatif():
    st.title("What-If Simulator")
    st.write(
        "Set a savings goal, then change income, spending or how much you save, and see how "
        "long the goal takes. Amounts are in rupees."
    )

    st.subheader("Your situation and goal")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        income = st.number_input("Monthly income (₹)", min_value=0, value=60000, step=1000)
        expenses = st.number_input("Monthly living expenses, excluding EMI (₹)", min_value=0, value=30000, step=1000)
        emi = st.number_input("Monthly EMI (₹)", min_value=0, value=8000, step=500)
    with c2:
        goal = st.number_input("Goal amount (₹)", min_value=1000, value=500000, step=10000)
        current = st.number_input("Already saved towards this goal (₹)", min_value=0, value=50000, step=5000)
        allocation = st.slider("Share of monthly surplus put towards the goal (%)", 0, 100, 70)
    annual_return = st.slider(
        "Expected yearly return (%)", 0.0, 15.0, 6.0, step=0.5,
        help="Enter a return after inflation if you want the goal read in today's rupees.",
    )

    st.subheader("What if...?")
    w1, w2 = st.columns(2, gap="large")
    with w1:
        d_income = st.slider("Change in income (%)", -50, 100, 10)
        d_expense = st.slider("Change in living expenses (%)", -50, 50, 0)
    with w2:
        d_alloc = st.slider("Change in share of surplus saved (percentage points)", -50, 50, 0)
        d_return = st.slider("Change in expected return (percentage points)", -5.0, 5.0, 0.0, step=0.5)

    scn_income = income * (1 + d_income / 100)
    scn_expenses = expenses * (1 + d_expense / 100)
    scn_alloc = min(max(allocation + d_alloc, 0), 100)
    scn_return = max(annual_return + d_return, 0.0)

    base = run_scenario(income, expenses, emi, allocation / 100, annual_return / 100, current, goal)
    scn = run_scenario(scn_income, scn_expenses, emi, scn_alloc / 100, scn_return / 100, current, goal)

    if base["surplus"] <= 0:
        st.warning("With today's numbers, expenses and EMI use up all of your income, so nothing is left to save.")
    if scn["surplus"] <= 0:
        st.warning("In the what-if plan, expenses and EMI use up all of your income, so nothing is left to save.")

    st.subheader("Results")
    r1, r2, r3 = st.columns(3)
    r1.metric("Time to goal: today's plan", fmt_duration(base["months"]))
    delta = None
    if base["months"] is not None and scn["months"] is not None:
        delta = f"{scn['months'] - base['months']:+d} months"
    r2.metric("Time to goal: what-if plan", fmt_duration(scn["months"]), delta=delta, delta_color="inverse")
    r3.metric(
        "Saved towards goal each month (what-if)",
        f"₹{scn['contribution']:,.0f}",
        delta=fmt_money_delta(scn["contribution"] - base["contribution"]),
    )

    if base["months"] is None and scn["months"] is None:
        st.error("Neither plan reaches the goal within 50 years. Try a larger surplus, a smaller goal or a higher return.")
    elif base["months"] is None:
        st.success(f"The what-if plan makes the goal reachable, in {fmt_duration(scn['months'])}.")
    elif scn["months"] is None:
        st.error("The what-if plan no longer reaches the goal within 50 years.")
    else:
        diff = base["months"] - scn["months"]
        if diff > 0:
            st.success(f"The what-if plan reaches the goal {fmt_duration(diff)} sooner.")
        elif diff < 0:
            st.warning(f"The what-if plan reaches the goal {fmt_duration(-diff)} later.")
        else:
            st.info("These changes make no difference to the time needed.")

    candidates = [m for m in (base["months"], scn["months"]) if m]
    horizon = int(max(candidates) * 1.2) + 6 if candidates else 120
    horizon = min(max(horizon, 60), 600)

    base_bal = project_balance(current, base["contribution"], annual_return / 100, horizon)
    scn_bal = project_balance(current, scn["contribution"], scn_return / 100, horizon)

    chart_df = pd.DataFrame({
        "Years": [m / 12 for m in range(horizon + 1)],
        "Today's plan": base_bal,
        "What-if plan": scn_bal,
    }).melt("Years", var_name="Plan", value_name="Balance")

    lines = alt.Chart(chart_df).mark_line().encode(
        x=alt.X("Years:Q", title="Years from now"),
        y=alt.Y("Balance:Q", title="Savings towards goal (₹)"),
        color=alt.Color("Plan:N", title=None),
        tooltip=[
            alt.Tooltip("Years:Q", format=".1f"),
            "Plan:N",
            alt.Tooltip("Balance:Q", format=",.0f"),
        ],
    )
    goal_line = alt.Chart(pd.DataFrame({"Goal": [goal]})).mark_rule(
        color="gray", strokeDash=[6, 4]
    ).encode(y="Goal:Q")
    st.altair_chart(lines + goal_line)
    st.caption(
        f"The dashed line is your goal. Balance after 5 years: ₹{base_bal[60]:,.0f} "
        f"(today's plan) vs ₹{scn_bal[60]:,.0f} (what-if plan)."
    )

    with st.expander("How this is calculated"):
        st.markdown(
            """
- Monthly surplus = income - living expenses - EMI.
- Amount saved each month = surplus x the share you put towards the goal.
- Each month the balance grows by the expected return (yearly rate divided by 12), then the monthly amount is added.
- Income, expenses and EMI are assumed constant. Taxes, market ups and downs and inflation are not modelled, so enter a return after inflation to read results in today's rupees.
- This is an illustration based on your inputs, not a prediction or financial advice.
"""
        )


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
| What-If Simulator | User inputs only | Month-by-month compound-growth projection towards a savings goal |
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
- The What-If Simulator assumes constant income, expenses and return, and does not model taxes, inflation or market risk.
- This is a decision-support tool, not financial advice.
"""
    )


# ---------- navigation ----------
PAGES = {
    "Overview": page_overview,
    "My Financial Health": page_health,
    "What-If Simulator": page_whatif,
    "Spending Forecast": page_forecast,
    "Customer Segments": page_segments,
    "About & Methodology": page_about,
}

st.sidebar.title("FinanceTwin")
choice = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("B.Tech Data Science · Interdisciplinary Project")

PAGES[choice]()