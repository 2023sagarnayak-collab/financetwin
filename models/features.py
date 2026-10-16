import numpy as np
import pandas as pd

CLASSES = ["Stable", "Stretched", "Distressed"]
EMPLOYMENT_TYPES = ["salaried", "self_employed", "informal"]

RAW_COLUMNS = [
    "monthly_salary", "annual_income", "monthly_expenses", "monthly_emi",
    "outstanding_debt", "loan_count", "credit_utilization_pct", "emergency_fund_months",
    "monthly_investment", "dependents", "city_tier", "age", "employment_type",
]

FEATURES = [
    "log_salary", "emi_to_salary", "expense_to_salary", "debt_to_income",
    "credit_utilization", "emergency_fund_months", "investment_to_salary",
    "dependents", "is_metro", "age", "loan_count", "is_self_employed", "is_informal",
    "surplus_k",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    s = df["monthly_salary"]
    out = pd.DataFrame(index=df.index)
    out["log_salary"] = np.log(s)
    out["emi_to_salary"] = df["monthly_emi"] / s
    out["expense_to_salary"] = df["monthly_expenses"] / s
    out["debt_to_income"] = df["outstanding_debt"] / df["annual_income"]
    out["credit_utilization"] = df["credit_utilization_pct"] / 100
    out["emergency_fund_months"] = df["emergency_fund_months"]
    out["investment_to_salary"] = df["monthly_investment"] / s
    out["dependents"] = df["dependents"]
    out["is_metro"] = (df["city_tier"] == 1).astype(int)
    out["age"] = df["age"]
    out["loan_count"] = df["loan_count"]
    out["is_self_employed"] = (df["employment_type"] == "self_employed").astype(int)
    out["is_informal"] = (df["employment_type"] == "informal").astype(int)
    # Absolute rupee surplus after expenses and EMI, in units of Rs 10,000
    out["surplus_k"] = (s - df["monthly_expenses"] - df["monthly_emi"]) / 10000
    return out