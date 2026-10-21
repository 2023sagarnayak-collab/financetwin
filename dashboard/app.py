import json
import math
import os
import sys

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models.features import build_features, FEATURES, RAW_COLUMNS, CLASSES, EMPLOYMENT_TYPES

DATA_PATH = os.path.join(ROOT, "data", "processed", "india_households.csv")
ART_DIR = os.path.join(ROOT, "models", "artifacts")

st.set_page_config(page_title="FinanceTwin", page_icon="📊", layout="wide")

# Plain-language names for the model's three classes
DISPLAY = {"Stable": "On track", "Stretched": "Getting tight", "Distressed": "Under strain"}

# Plain-language description of each factor
LABELS = {
    "debt_to_income": "Your total loans compared with your yearly income",
    "emi_to_salary": "Share of your salary going to loan payments",
    "credit_utilization": "Share of your credit card limit you are using",
    "expense_to_salary": "Share of your salary going on everyday spending",
    "emergency_fund_months": "Savings you could live on (months of spending)",
    "investment_to_salary": "Share of your salary you invest",
    "surplus_k": "Money left at the end of the month",
    "is_informal": "Informal or daily-wage work",
    "is_self_employed": "Self-employed",
    "is_metro": "Living in a big metro city",
    "log_salary": "Your monthly salary",
    "loan_count": "Number of loans",
    "dependents": "People who rely on your income",
    "age": "Your age",
}

RATIO_FEATURES = {"debt_to_income", "emi_to_salary", "expense_to_salary",
                  "investment_to_salary", "credit_utilization"}
YES_NO_FEATURES = {"is_informal", "is_self_employed", "is_metro"}
HIGHER_IS_WORSE = {"debt_to_income", "emi_to_salary", "credit_utilization", "expense_to_salary", "is_informal"}
LOWER_IS_WORSE = {"emergency_fund_months", "investment_to_salary", "surplus_k"}

ADVICE = {
    "debt_to_income": "Your loans are large compared with your yearly income. Paying off the most expensive loan first saves the most.",
    "emi_to_salary": "A large share of your salary goes to loan payments. Ask your lender about lowering the rate or combining loans.",
    "credit_utilization": "You are using much of your credit card limit. Keeping card balances below 30% of the limit helps.",
    "expense_to_salary": "Everyday spending takes most of your salary. Start by reviewing fixed costs such as subscriptions and bills.",
    "emergency_fund_months": "Your savings would last only a few months. Aim for 3 to 6 months of spending in a separate account.",
    "investment_to_salary": "You invest only a small part of your salary. A small regular amount is a good start once savings are in place.",
    "surplus_k": "Very little is left after spending and loan payments. Even a small cut in one bill would make a difference.",
}


# ---------- loading ----------
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(ART_DIR, "health_model.joblib"))


@st.cache_data
def load_metrics():
    with open(os.path.join(ART_DIR, "metrics.json")) as f:
        return json.load(f)


@st.cache_data
def load_group_report():
    return pd.read_csv(os.path.join(ART_DIR, "group_report.csv"))


@st.cache_data
def load_error_diagnosis():
    return pd.read_csv(os.path.join(ART_DIR, "group_error_diagnosis.csv"))


@st.cache_data
def load_stable_medians():
    raw = pd.read_csv(DATA_PATH)
    feats = build_features(raw)[FEATURES]
    stable = raw["health_class"] == "Stable"
    borrower = raw["monthly_emi"] > 0
    med = feats[stable].median()
    # Loan factors are compared with "on track" households that have a loan
    loan_cols = ["debt_to_income", "emi_to_salary"]
    med[loan_cols] = feats.loc[stable & borrower, loan_cols].median()
    return med


# ---------- model calls ----------
def predict(profile: dict):
    model = load_model()
    X = build_features(pd.DataFrame([profile])[RAW_COLUMNS])[FEATURES]
    raw = model.predict_proba(X)[0]
    order = list(model.classes_)
    probs = {c: float(raw[order.index(c)]) for c in CLASSES}
    return max(probs, key=probs.get), probs, X.iloc[0]


def is_worse(feature, value, typical, tolerance=0.10):
    if feature in HIGHER_IS_WORSE:
        return value > typical * (1 + tolerance)
    if feature in LOWER_IS_WORSE:
        return value < typical * (1 - tolerance)
    return False


def fmt_value(feature, value):
    """Show each factor in the unit a user would recognise."""
    if feature in RATIO_FEATURES:
        return f"{value:.0%}"
    if feature == "emergency_fund_months":
        return f"{value:.1f} months"
    if feature == "surplus_k":
        return f"₹{value * 10000:,.0f}"
    if feature == "log_salary":
        return f"₹{math.exp(value):,.0f}"
    if feature in YES_NO_FEATURES:
        return "Yes" if value >= 0.5 else "No"
    return f"{value:g}"


def driver_table(x, top_n=4):
    stable = load_stable_medians()
    importance = load_metrics()["importance"]
    top = [f for f in importance if f in LABELS][:top_n]
    rows = []
    for f in top:
        rows.append({
            "What it measures": LABELS[f],
            "Your figure": fmt_value(f, x[f]),
            "Typical 'on track' household": fmt_value(f, stable[f]),
            "Assessment": "Needs attention" if is_worse(f, x[f], stable[f]) else "Looks fine",
        })
    return pd.DataFrame(rows)


def advice(x):
    stable = load_stable_medians()
    msgs = [ADVICE[f] for f in ADVICE if is_worse(f, x[f], stable[f])]
    return msgs or ["Nothing stands out. Keep checking in when your situation changes."]


def prob_chart(probs):
    df = pd.DataFrame({
        "Result": [DISPLAY[c] for c in CLASSES],
        "Chance": [probs[c] for c in CLASSES],
    })
    return alt.Chart(df).mark_bar(cornerRadius=4).encode(
        x=alt.X("Chance:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Chance")),
        y=alt.Y("Result:N", sort=[DISPLAY[c] for c in CLASSES], title=None),
        tooltip=["Result", alt.Tooltip("Chance:Q", format=".0%")],
    ).properties(height=160)


EXPENSE_CUTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
EMI_CUTS = [0.0, 0.10, 0.20, 0.30, 0.50]
SAVING_ADDS = [0, 1, 2, 3, 6]


def path_to_stable(profile, current_class):
    """Find the smallest set of changes the model rates 'On track'.
    Rule of thumb for 'smallest': 1 point per 1% cut in spending or loan payments, 5 points per month of savings."""
    if current_class == "Stable":
        return {"already_stable": True}

    grid = [(ec, mc, sa) for ec in EXPENSE_CUTS for mc in EMI_CUTS for sa in SAVING_ADDS]
    rows = []
    for ec, mc, sa in grid:
        p = dict(profile)
        p["monthly_expenses"] = profile["monthly_expenses"] * (1 - ec)
        p["monthly_emi"] = profile["monthly_emi"] * (1 - mc)
        p["emergency_fund_months"] = profile["emergency_fund_months"] + sa
        rows.append(p)

    model = load_model()
    X = build_features(pd.DataFrame(rows)[RAW_COLUMNS])[FEATURES]
    proba = model.predict_proba(X)
    stable_col = list(model.classes_).index("Stable")
    stable_p = proba[:, stable_col]
    new_class = [model.classes_[i] for i in proba.argmax(axis=1)]

    cost = [ec * 100 + mc * 100 + sa * 5 for ec, mc, sa in grid]
    stable_idx = [i for i in range(len(grid)) if new_class[i] == "Stable"]

    if stable_idx:
        best = min(stable_idx, key=lambda i: (cost[i], -stable_p[i]))
        found = True
    else:
        best = int(stable_p.argmax())
        found = False

    ec, mc, sa = grid[best]
    return {
        "already_stable": False,
        "found": found,
        "expense_cut": ec,
        "emi_cut": mc,
        "savings_months": sa,
        "stable_probability": float(stable_p[best]),
        "new_class": new_class[best],
    }


def months_to_target(profile, target_months, share):
    exp = profile["monthly_expenses"]
    surplus = profile["monthly_salary"] - exp - profile["monthly_emi"]
    gap = max(target_months * exp - profile["emergency_fund_months"] * exp, 0)
    if gap == 0:
        return 0
    contribution = max(surplus, 0) * share
    if contribution <= 0:
        return None
    return math.ceil(gap / contribution)


# ---------- pages ----------
def page_home():
    st.title("FinanceTwin")
    st.subheader("Understand your household's money health in a few minutes")
    st.write(
        "Most budgeting apps only show where your money went. FinanceTwin tells you whether your "
        "finances are on track, getting tight, or under strain, explains what is behind that result, "
        "and suggests what could help."
    )

    st.subheader("How to use it")
    a, b, c = st.columns(3)
    with a.container(border=True):
        st.markdown("**1. Enter your figures**")
        st.write("Type in your pay, spending, loan payments and savings. Each box explains what to enter.")
    with b.container(border=True):
        st.markdown("**2. See your result**")
        st.write("You get one of three results, with the main reasons behind it.")
    with c.container(border=True):
        st.markdown("**3. Try changes**")
        st.write("Test what would happen if you cut spending, lowered a loan payment or saved more.")

    st.subheader("What you get")
    st.write(
        "- A simple result: **On track**, **Getting tight** or **Under strain**\n"
        "- The main reasons for your result, in plain words\n"
        "- The simplest changes that could move you to **On track**\n"
        "- A timeline for building a safety cushion of savings"
    )

    m = load_metrics()
    acc = m["models"][m["selected"]]["test_accuracy"]
    st.info(
        f"**Please note:** this tool was built and tested on made-up households, not real people. "
        f"In tests it classed about {round(acc * 100)} in every 100 of them correctly. It gives "
        "an estimate to help you think, not financial advice."
    )


def page_check():
    st.title("Check your money health")
    st.write("Fill in your details on the left. Your result and tips appear on the right. Amounts are per month, in rupees.")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Your details")
        salary = st.slider(
            "Monthly take-home pay (₹)", 10000, 400000, 40000, step=1000,
            help="The salary you actually receive each month, after tax and deductions.",
        )
        employment = st.selectbox(
            "Type of work", EMPLOYMENT_TYPES,
            format_func=lambda e: {"salaried": "Salaried (fixed monthly pay)",
                                   "self_employed": "Self-employed or business",
                                   "informal": "Informal or daily wage"}[e],
            help="Choose the option that best describes how you earn.",
        )
        loans = st.slider(
            "Number of loans you are repaying", 0, 3, 1,
            help="Home, personal, vehicle or card loans. Choose 0 if you have none.",
        )
        expenses = st.slider(
            "Monthly everyday spending (₹)", 0, 250000, 22000, step=1000,
            help="Food, rent, travel, bills and school fees. Do not include loan payments.",
        )
        emi = st.slider(
            "Total monthly loan payments, called EMI (₹)", 0, 150000, 8000, step=500,
            help="All your monthly loan instalments added together.",
        )
        debt = st.slider(
            "Total still owed on all loans (₹)", 0, 5000000, 300000, step=10000,
            help="The amount you still have to repay, not the monthly payment.",
        )
        util = st.slider(
            "Credit card limit used (%)", 0, 100, 25,
            help="Your card balance divided by your card limit. 25% means you use a quarter of the limit.",
        )
        emergency = st.slider(
            "Savings you could live on (months)", 0.0, 24.0, 3.0, step=0.5,
            help="Money in savings or easy-to-access funds, divided by your monthly spending. "
                 "3 means your savings would cover three months of spending.",
        )
        invest = st.slider(
            "Amount invested each month (₹)", 0, 100000, 4000, step=500,
            help="Money you put into investments such as mutual funds, PPF or fixed deposits.",
        )
        age = st.slider("Your age", 21, 59, 35)
        dependents = st.slider(
            "People who rely on your income", 0, 4, 2,
            help="Children, parents or others who depend on your earnings.",
        )
        tier = st.radio(
            "Where do you live?", [1, 2, 3], index=1, horizontal=True,
            format_func=lambda t: {1: "Big metro", 2: "Large city", 3: "Small town"}[t],
            help="Big cities usually have higher rent and everyday costs. Your monthly spending figure should include those costs.",
        )

    profile = {
        "monthly_salary": salary,
        "annual_income": salary * 12,
        "monthly_expenses": expenses,
        "monthly_emi": emi,
        "outstanding_debt": debt,
        "credit_utilization_pct": util,
        "emergency_fund_months": emergency,
        "monthly_investment": invest,
        "dependents": dependents,
        "city_tier": tier,
        "age": age,
        "loan_count": loans,
        "employment_type": employment,
    }
    cls, probs, x = predict(profile)

    with right:
        st.subheader("Your result")
        st.metric("Money health", DISPLAY[cls])
        st.altair_chart(prob_chart(probs), use_container_width=True)

        if cls == "Stable":
            st.success("You look on track. Keep your savings and loan payments steady.")
        elif cls == "Stretched":
            st.warning("Your finances are getting tight. Small changes now can prevent bigger problems later.")
        else:
            st.error(
                "Your finances look under strain. Consider talking to a financial adviser or a "
                "non-profit credit counsellor."
            )

        st.subheader("What is affecting your result")
        st.table(driver_table(x).set_index("What it measures"))

        st.subheader("What you could do")
        for msg in advice(x):
            st.info(msg)

        st.caption("An estimate from a model trained on made-up households. Not financial advice.")

    st.divider()
    st.subheader("Try a change")
    st.caption("Move a slider to see how a change could affect your result. Your original figures stay as they are.")
    w1, w2, w3, w4 = st.columns(4)
    d_sal = w1.slider("Change my take-home pay by (%)", -50, 50, 0)
    d_exp = w2.slider("Change my spending by (%)", -50, 50, 0)
    d_emi = w3.slider("Change my loan payments by (₹)", -20000, 20000, 0, step=500)
    d_sav = w4.slider("Change my savings by (months)", -3.0, 6.0, 0.0, step=0.5)

    scenario = dict(
        profile,
        monthly_salary=salary * (1 + d_sal / 100),
        annual_income=salary * (1 + d_sal / 100) * 12,
        monthly_expenses=expenses * (1 + d_exp / 100),
        monthly_emi=max(emi + d_emi, 0),
        emergency_fund_months=max(emergency + d_sav, 0.0),
    )
    s_cls, s_probs, _ = predict(scenario)

    order = {c: i for i, c in enumerate(CLASSES)}
    steps = order[s_cls] - order[cls]
    if steps == 0:
        result_delta = None
    elif steps > 0:
        result_delta = f"Worse: {steps} step{'s' if steps > 1 else ''} down from {DISPLAY[cls]}"
    else:
        result_delta = f"Better: {-steps} step{'s' if steps < -1 else ''} up from {DISPLAY[cls]}"

    dd = (s_probs["Distressed"] - probs["Distressed"]) * 100
    strain_delta = None if round(dd) == 0 else f"{dd:+.0f} points"

    c1, c2 = st.columns(2)
    c1.metric("Money health after this change", DISPLAY[s_cls], delta=result_delta)
    c2.metric(
        "Chance of being under strain",
        f"{s_probs['Distressed']:.0%}",
        delta=strain_delta,
        delta_color="inverse",
    )

    st.divider()
    st.subheader("Simplest steps to get to 'On track'")
    st.caption(
        "Based on the scenario above (the same as your figures when the sliders are at zero). "
        "Lowering loan payments assumes refinancing or combining loans, which cuts the monthly "
        "amount without changing what you owe."
    )

    plan = path_to_stable(scenario, s_cls)
    if plan["already_stable"]:
        st.success("You are already on track. No change is needed, but keep your savings growing.")
    else:
        steps_text = []
        if plan["expense_cut"] > 0:
            steps_text.append(f"cut everyday spending by {plan['expense_cut']:.0%}")
        if plan["emi_cut"] > 0:
            steps_text.append(f"lower monthly loan payments by {plan['emi_cut']:.0%}")
        if plan["savings_months"] > 0:
            steps_text.append(f"build savings up by {plan['savings_months']} month(s) of spending")
        joined = " and ".join(steps_text)
        if plan["found"]:
            st.info(
                f"The simplest combination that could move you to **On track**: {joined}. "
                f"The estimated chance of being on track rises to {plan['stable_probability']:.0%}."
            )
        else:
            st.warning(
                "No combination in the range we tested reaches On track. The best option, "
                f"{joined}, would move you to {DISPLAY[plan['new_class']]} "
                f"({plan['stable_probability']:.0%} chance of being on track). Bigger changes, "
                "or a higher income, would be needed."
            )

    st.subheader("How long to build a safety cushion")
    f1, f2 = st.columns(2)
    share = f1.slider(
        "Share of your monthly leftover you can set aside (%)", 10, 100, 50, step=5,
        help="Leftover means pay minus spending minus loan payments.",
    )
    target = f2.slider(
        "Savings goal (months of spending)", 3, 12, 6,
        help="Three to six months of spending is a common safety-cushion goal.",
    )
    months = months_to_target(scenario, target, share / 100)
    if months is None:
        st.warning("You have no money left at the end of the month, so savings cannot grow yet. Lower spending or loan payments first.")
    elif months == 0:
        st.success(f"You already have at least {target} months of spending saved. Well done.")
    else:
        st.info(
            f"Setting aside {share}% of your leftover each month, you reach {target} months of "
            f"savings in about {months} months ({months / 12:.1f} years)."
        )


def page_how_it_works():
    st.title("How it works")
    m = load_metrics()
    groups_all = load_group_report()
    groups = groups_all[groups_all["model"] == m["selected"]]
    acc = m["models"][m["selected"]]["test_accuracy"]

    st.markdown(
        """
### In simple words
FinanceTwin looks at the same things a bank would: how much you earn, what you spend, how big your loan payments are and how much you could fall back on. It then places your household in one of three groups:

- **On track:** your income covers your spending and loans, with room to save.
- **Getting tight:** there is some pressure, but you still have room to act.
- **Under strain:** repayments and spending leave little margin, so a small shock could cause problems.

### How accurate is it?
In tests on made-up households, about **{round(acc * 100)} in every 100** were placed in the right group. Almost all mistakes were between neighbouring groups, such as **On track** and **Getting tight**. Mistakes that jumped from **On track** straight to **Under strain** were very rare.

### What it cannot do
- It was built and tested on **made-up households**, so it shows how the method works, not how it would perform on real people.
- It does not know the interest rate on your loans, your tax position or your long-term goals.
- It is not financial advice. For decisions about loans, investments or debt, speak to a qualified adviser.
"""
    )

    with st.expander("Words explained"):
        st.markdown(
            """
- **EMI:** the fixed monthly instalment you pay on a loan.
- **Credit card limit used:** your card balance divided by your card limit.
- **Savings you could live on:** your easy-to-access savings divided by your monthly spending.
- **Leftover:** what remains after spending and loan payments.
- **Refinancing:** replacing a loan with a new one at a lower rate or longer term, which usually lowers the monthly payment.
- **Safety cushion:** savings that cover several months of spending, for emergencies.
"""
        )

    with st.expander("Technical details (for readers who want them)"):
        st.write(
            "The model is a supervised classifier trained on 150,000 synthetic households, with the "
            "label built from burden relative to income and the rupee leftover. Two models were "
            "compared with five-fold cross-validation, and the held-out test set was used once."
        )
        rows = {}
        for name, r in m["models"].items():
            rows[name] = {
                "Cross-validated macro-F1": f"{r['cv_macro_f1_mean']:.3f} ± {r['cv_macro_f1_std']:.3f}",
                "Test accuracy": r["test_accuracy"],
                "Test macro-F1": r["test_macro_f1"],
            }
        st.table(pd.DataFrame(rows).T)
        st.write(f"Selected model: **{m['selected']}**. {m['selection_reason']}")

        st.write("**Confusion matrix on the test set** (rows: actual, columns: predicted)")
        cm = pd.DataFrame(m["confusion_matrix"], index=m["class_order"], columns=m["class_order"])
        st.table(cm)

        st.write("**Accuracy by household type (test set)**")
        st.table(groups.drop(columns="model").set_index(["dimension", "group"]))

        st.write("**Where errors fall (training split)**")
        st.table(load_error_diagnosis().set_index(["dimension", "group"])[
            ["n", "accuracy", "errors", "neighbour_error_share", "stable_vs_distressed_errors"]
        ])

    st.markdown(
        """
### Our assumptions
Financial health is defined in two ways in this tool: how much of your income goes on loan payments and debts, and how much is left at the end of the month. Both are our assumptions about what matters. Real data would show whether they predict real outcomes.
"""
    )


# ---------- navigation ----------
PAGES = {
    "Home": page_home,
    "Check my money health": page_check,
    "How it works": page_how_it_works,
}

st.sidebar.title("FinanceTwin")
choice = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("B.Tech Data Science · Interdisciplinary Project")

PAGES[choice]()