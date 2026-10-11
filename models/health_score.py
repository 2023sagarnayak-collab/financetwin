import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the ratio-based indicators the Health Score depends on."""
    df = df.copy()

    df["debt_to_income"] = df["Outstanding_Debt"] / df["Annual_Income"]
    df["savings_rate"] = df["Monthly_Balance"] / df["Monthly_Inhand_Salary"]
    df["investment_ratio"] = df["Amount_invested_monthly"] / df["Monthly_Inhand_Salary"]
    df["emi_burden"] = df["Total_EMI_per_month"] / df["Monthly_Inhand_Salary"]
    df["credit_utilization"] = df["Credit_Utilization_Ratio"] / 100  # already 0-100 scale

    # Monthly_Inhand_Salary has missing values (~15% of rows) — drop rows
    # where we can't compute the core ratios rather than guessing a fill value.
    df = df.dropna(subset=["debt_to_income", "savings_rate", "investment_ratio", "emi_burden"])

    return df


def normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max scale a feature to 0-1. If invert=True, higher raw value = worse (lower score)."""
    clipped = series.clip(series.quantile(0.01), series.quantile(0.99))
    scaled = (clipped - clipped.min()) / (clipped.max() - clipped.min())
    return 1 - scaled if invert else scaled


def compute_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted Health Score (0-100).
    Weights are a documented design choice, not learned — this is the
    transparent baseline version, before any ML model.
    """
    df = df.copy()

    df["score_debt"] = normalize(df["debt_to_income"], invert=True)       # lower debt/income = better
    df["score_savings"] = normalize(df["savings_rate"], invert=False)     # higher savings rate = better
    df["score_investment"] = normalize(df["investment_ratio"], invert=False)
    df["score_emi"] = normalize(df["emi_burden"], invert=True)            # lower EMI burden = better
    df["score_credit_util"] = normalize(df["credit_utilization"], invert=True)

    # TODO: savings_rate, investment_ratio, and credit_utilization showed
    # weak/inverted correlation with actual Credit_Score in this dataset,
    # likely because payment-history features (Num_of_Delayed_Payment,
    # Delay_from_due_date, Payment_of_Min_Amount) are the dataset's true
    # drivers but aren't yet incorporated. Revisit with those features
    # for a stronger v2 model.
    weights = {
        "score_debt": 0.45,
        "score_emi": 0.35,
        "score_savings": 0.10,
        "score_investment": 0.05,
        "score_credit_util": 0.05,
    }

    df["health_score"] = sum(df[col] * w for col, w in weights.items()) * 100
    return df


if __name__ == "__main__":
    credit = pd.read_csv(os.path.join(PROCESSED_DIR, "credit_clean.csv"))
    credit = engineer_features(credit)
    credit = compute_health_score(credit)

    print(f"Rows scored: {len(credit)}")
    print(credit["health_score"].describe())
    print("\nSample scores:")
    print(credit[["Annual_Income", "debt_to_income", "savings_rate", "health_score"]].sample(5))


    print("\nMean health_score by actual Credit_Score label:")
    print(credit.groupby("Credit_Score")["health_score"].mean())

    print("\nMean of each sub-score by actual Credit_Score label:")
    print(credit.groupby("Credit_Score")[
        ["score_debt", "score_savings", "score_investment", "score_emi", "score_credit_util"]
    ].mean())

    print("\nRaw feature means by actual Credit_Score label:")
    print(credit.groupby("Credit_Score")[
        ["debt_to_income", "savings_rate", "investment_ratio", "emi_burden", "credit_utilization"]
    ].mean())