import json
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

LABELS = {
    "debt_to_income": "Debt-to-income (loan balance / yearly income)",
    "emi_to_salary": "EMI burden (EMI / salary, borrowers)",
    "credit_utilization": "Credit card utilisation",
    "expense_to_salary": "Living expenses / salary",
    "emergency_fund_months": "Emergency savings (months of expenses)",
    "investment_to_salary": "Investment / salary",
    "dependents": "Dependents",
    "is_metro": "Metro city (Tier 1)",
    "age": "Age",
    "log_salary": "Monthly salary (log scale)",
    "loan_count": "Number of loans",
    "is_self_employed": "Self-employed",
    "is_informal": "Informal or daily-wage work",
    "surplus_k": "Monthly surplus after expenses and EMI (₹10,000s)",
}

HIGHER_IS_WORSE = {"debt_to_income", "emi_to_salary", "credit_utilization", "expense_to_salary", "is_informal"}
LOWER_IS_WORSE = {"emergency_fund_months", "investment_to_salary", "surplus_k"}

ADVICE = {
    "debt_to_income": "Your loan balance is high relative to your yearly income. Prioritise repaying the most expensive loan first.",
    "emi_to_salary": "EMIs take a large share of your salary. Look at refinancing or prepaying the costliest loan.",
    "credit_utilization": "You are using a large share of your credit limit. Bringing card balances below 30% helps.",
    "expense_to_salary": "Living costs take most of your salary. Review fixed costs first.",
    "emergency_fund_months": "Your emergency savings cover few months of expenses. A common target is 3 to 6 months.",
    "investment_to_salary": "Little is being invested. A small regular investment can start once the emergency fund is in place.",
    "surplus_k": "Little is left after expenses and EMIs each month. Cutting one fixed cost or one EMI would widen your cushion.",
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
    # Loan factors are compared with Stable households that have a loan
    loan_cols = ["debt_to_income", "emi_to_salary"]
    med[loan_cols] = feats.loc[stable & borrower, loan_cols].median()
    return med


# ---------- model call ----------
def predict(profile: dict):
    """One entry point for every part of the page: profile in, class and probabilities out."""
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


def driver_table(x, top_n=4):
    stable = load_stable_medians()
    importance = load_metrics()["importance"]
    top = [f for f in importance if f in FEATURES][:top_n]
    rows = []
    for f in top:
        rows.append({
            "Factor": LABELS[f],
            "Your value": round(float(x[f]), 2),
            "Typical Stable household": round(float(stable[f]), 2),
            "Status": "Worse than typical" if is_worse(f, x[f], stable[f]) else "In line",
        })
    return pd.DataFrame(rows)


def advice(x):
    stable = load_stable_medians()
    msgs = [ADVICE[f] for f in FEATURES if f in ADVICE and is_worse(f, x[f], stable[f])]
    return msgs or ["No factor is worse than a typical Stable household. Keep checking as your situation changes."]


def prob_chart(probs):
    df = pd.DataFrame({"Class": CLASSES, "Probability": [probs[c] for c in CLASSES]})
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("Class:N", sort=CLASSES, title=None),
        tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
    )


# ---------- pages ----------
def page_overview():
    st.title("FinanceTwin")
    st.subheader("A supervised model for household financial health in India")
    st.write(
        "Enter one household profile. A trained model predicts whether the household is Stable, "
        "Stretched or Distressed, explains which factors drive that prediction, and shows how "
        "changes would shift it."
    )
    m = load_metrics()
    a, b, c = st.columns(3)
    a.metric("Households in training data", f"{m['train_rows'] + m['test_rows']:,}")
    b.metric("Selected model", m["selected"])
    c.metric("Macro-F1 on held-out test set", f"{m['models'][m['selected']]['test_macro_f1']:.2f}")
    st.info(
        "The training data is synthetic, built from assumptions about Indian households. Accuracy "
        "shows how well the method recovers the labelling rule, not performance on real "
        "households. See About & Evidence."
    )


def page_check():
    st.title("Financial Check")
    st.write("Enter your household figures in rupees. The same model answers every question on this page.")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Your household")
        salary = st.slider("Monthly in-hand salary (₹)", 10000, 400000, 40000, step=1000)
        employment = st.selectbox("Type of work", EMPLOYMENT_TYPES,
                                  format_func=lambda e: e.replace("_", " ").capitalize())
        loans = st.slider("Number of loans", 0, 3, 1)
        expenses = st.slider("Monthly living expenses, excluding EMI (₹)", 0, 250000, 22000, step=1000)
        emi = st.slider("Total EMI per month (₹)", 0, 150000, 8000, step=500)
        debt = st.slider("Outstanding loan balance (₹)", 0, 5000000, 300000, step=10000)
        util = st.slider("Credit card utilisation (%)", 0, 100, 25)
        emergency = st.slider("Emergency savings (months of expenses)", 0.0, 24.0, 3.0, step=0.5)
        invest = st.slider("Invested per month (₹)", 0, 100000, 4000, step=500)
        age = st.slider("Age", 21, 59, 35)
        dependents = st.slider("Dependents", 0, 4, 2)
        tier = st.radio("City tier", [1, 2, 3], index=1, horizontal=True,
                        format_func=lambda t: f"Tier {t}")
        st.caption("Results are most reliable within the ranges of the training data.")

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
        st.subheader("Prediction")
        st.metric("Predicted financial-health class", cls)
        st.altair_chart(prob_chart(probs))
        if cls == "Stable":
            st.success("This household's profile looks Stable.")
        elif cls == "Stretched":
            st.warning("This household's profile looks Stretched: some pressure, but not yet critical.")
        else:
            st.error("This household's profile looks Distressed: repayments and savings are under serious strain.")
        st.caption("Model-based estimate from synthetic data. Not financial advice.")

        st.subheader("What is driving this")
        st.table(driver_table(x))

        st.subheader("Suggestions")
        for msg in advice(x):
            st.info(msg)

    st.divider()
    st.subheader("What if...?")
    st.caption("The same model re-scores the household after each change.")
    w1, w2, w3, w4 = st.columns(4)
    d_sal = w1.slider("Change in salary (%)", -50, 50, 0)
    d_exp = w2.slider("Change in expenses (%)", -50, 50, 0)
    d_emi = w3.slider("Change in EMI (₹)", -20000, 20000, 0, step=500)
    d_sav = w4.slider("Change in emergency savings (months)", -3.0, 6.0, 0.0, step=0.5)

    scenario = dict(
        profile,
        monthly_salary=salary * (1 + d_sal / 100),
        annual_income=salary * (1 + d_sal / 100) * 12,
        monthly_expenses=expenses * (1 + d_exp / 100),
        monthly_emi=max(emi + d_emi, 0),
        emergency_fund_months=max(emergency + d_sav, 0.0),
    )
    s_cls, s_probs, _ = predict(scenario)

    c1, c2 = st.columns(2)
    c1.metric("Predicted class (what-if)", s_cls, delta=None if s_cls == cls else f"was {cls}")
    c2.metric(
        "Chance of Distressed (what-if)",
        f"{s_probs['Distressed']:.0%}",
        delta=f"{(s_probs['Distressed'] - probs['Distressed']) * 100:+.0f} points",
        delta_color="inverse",
    )


def page_about():
    st.title("About & Evidence")
    m = load_metrics()
    groups_all = load_group_report()
    groups = groups_all[groups_all["model"] == m["selected"]]

    def group_value(dim, grp, col):
        return groups.loc[(groups["dimension"] == dim) & (groups["group"] == grp), col].iloc[0]

    informal_acc = group_value("employment", "informal", "accuracy")
    top_band_n = int(group_value("income band", "2L-4L", "n"))

    st.markdown(
        """
**FinanceTwin** is a supervised model that reads a household's finances and predicts whether it is Stable, Stretched or Distressed.

### How the model was built
1. **Data:** 150,000 synthetic Indian households. Salaries range from ₹10,000 to ₹4,00,000 a month, with salaried, self-employed and informal workers, zero to three loans, and three city tiers. The health label comes from a documented risk formula plus noise.
2. **Definition of financial health:** the label combines two measures. The first is burden relative to income: EMI, debt and credit use as shares of salary or income. The second is the rupee cushion left after expenses and EMI, because that cushion determines whether a household can absorb a shock. Both are our assumptions about what financial health means, not estimates from data.
3. **Features:** 14 ratios and attributes, built by one shared function that both training and the app use.
4. **Models:** two candidates, logistic regression and random forest.
5. **Selection:** both were scored with five-fold cross-validation on the training split only, using macro-F1.
6. **Final check:** the 20% held-out test set was used once, after selection.
"""
    )

    st.subheader("Model comparison")
    rows = {}
    for name, r in m["models"].items():
        rows[name] = {
            "CV macro-F1": f"{r['cv_macro_f1_mean']:.3f} ± {r['cv_macro_f1_std']:.3f}",
            "Test accuracy": r["test_accuracy"],
            "Test macro-F1": r["test_macro_f1"],
        }
    base = m["reference"]["Majority-class baseline"]
    rows["Majority-class baseline (reference only)"] = {
        "CV macro-F1": "n/a",
        "Test accuracy": base["test_accuracy"],
        "Test macro-F1": base["test_macro_f1"],
    }
    st.table(pd.DataFrame(rows).T)
    st.success(f"**Selected: {m['selected']}.** {m['selection_reason']}")

    st.subheader("Confusion matrix on the test set (rows: actual, columns: predicted)")
    cm = pd.DataFrame(m["confusion_matrix"], index=m["class_order"], columns=m["class_order"])
    st.table(cm)

    st.subheader("Performance by household type (test set, selected model)")
    st.caption("Accuracy is the more reliable measure for small groups. Macro-F1 is unstable when a group has few cases of one class.")
    st.table(groups.drop(columns="model").set_index(["dimension", "group"]))

    st.subheader("Where the errors fall (training split, cross-validated)")
    st.caption(
        "A separate analysis on the training split, so these figures are not the test-set results above. "
        "A neighbour error is one between adjacent classes (Stable and Stretched, or Stretched and Distressed)."
    )
    diag = load_error_diagnosis()
    st.table(diag.set_index(["dimension", "group"])[
        ["n", "accuracy", "errors", "neighbour_error_share", "stable_vs_distressed_errors"]
    ])

    st.markdown(
        f"""
### Income and the trade-off
We tested three definitions of financial health. A ratio-only label gave overall test accuracy of 0.871, with informal-worker accuracy of 0.826. A buffer penalty for low salaries gave 0.869 overall but reduced informal accuracy to 0.797. The surplus definition used here gave 0.877 overall, with informal accuracy of 0.813 and accuracy of 0.826 for households earning under ₹20,000. Surplus improves overall accuracy compared with the ratio-only model, but it costs some accuracy for informal and low-income households. We report this trade-off instead of choosing the setting that looks best overall.

### Limitations
- **Synthetic data.** High accuracy shows the model recovers the labelling rule, not that it predicts real households.
- **Definition is a choice.** The surplus term reflects our view of financial health. Real data would show whether it predicts outcomes.
- **Informal households.** Accuracy for informal workers is {informal_acc:.1%}. In the training-split analysis, about 24% of Stable informal households are flagged as Stretched, against about 8% of salaried households.
- **Boundary errors.** Almost all errors are between neighbouring classes. Only a handful are Stable versus Distressed.
- **Thin top income band.** The ₹2L to ₹4L band has {top_band_n} test households, so its scores are unreliable.
- **Decision support only.** This is a model-based estimate, not financial advice. Real validation needs household microdata, for example the MoSPI AIDIS 2019 unit-level data, and review by a qualified adviser.
"""
    )


# ---------- navigation ----------
PAGES = {
    "Overview": page_overview,
    "Financial Check": page_check,
    "About & Evidence": page_about,
}

st.sidebar.title("FinanceTwin")
choice = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("B.Tech Data Science · Interdisciplinary Project")

PAGES[choice]()