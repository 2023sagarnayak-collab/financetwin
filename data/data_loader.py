import pandas as pd
import numpy as np
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Strip stray non-numeric characters (e.g. trailing underscores) and convert to float."""
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def load_credit_data() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "train.csv"), low_memory=False)

    numeric_cols = [
        "Age", "Annual_Income", "Outstanding_Debt",
        "Num_of_Loan", "Changed_Credit_Limit", "Monthly_Balance",
        "Amount_invested_monthly",
    ]
    for col in numeric_cols:
        df[col] = clean_numeric_column(df[col])

    # Drop impossible ages
    df = df[(df["Age"] >= 0) & (df["Age"] <= 100)]

    # Catch placeholder/garbage sentinels (e.g. absurd repeated-digit values)
    # that survive numeric parsing but are not real financial figures.
    df.loc[df["Monthly_Balance"].abs() > 1e5, "Monthly_Balance"] = np.nan
    df.loc[df["Annual_Income"].abs() > 1e7, "Annual_Income"] = np.nan

    # Winsorize genuine extreme outliers (documented modeling choice,
    # not a data error) so a handful of very high earners don't dominate
    # ratio-based features like debt-to-income.
    income_cap = df["Annual_Income"].quantile(0.99)
    df["Annual_Income"] = df["Annual_Income"].clip(upper=income_cap)

    # same reasoning — cap extreme EMI outliers before using this
    # in any ratio-based feature
    emi_cap = df["Total_EMI_per_month"].quantile(0.99)
    df["Total_EMI_per_month"] = df["Total_EMI_per_month"].clip(upper=emi_cap)

    # Drop rows where core fields are still missing after cleaning
    df = df.dropna(subset=["Annual_Income", "Outstanding_Debt", "Age", "Monthly_Balance"])

    return df


def load_budget_transactions() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "budget_transactions.csv"))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["category"] = df["category"].str.strip().str.lower()
    # fix known typos from the source data
    df["category"] = df["category"].replace({
        "restuarant": "restaurant",
        "coffe": "coffee",
    })
    return df


def load_customer_transactions() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "customer_transactions.csv"))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def load_retail_transactions() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "online_retail_II.csv"))

    # Drop anonymous/guest purchases — can't attribute these to a customer
    df = df.dropna(subset=["Customer ID"])

    # Drop cancelled invoices (Invoice starts with 'C') and returns (negative quantity)
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[df["Quantity"] > 0]

    # Drop invalid prices
    df = df[df["Price"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    df["Customer ID"] = df["Customer ID"].astype(int)

    return df


if __name__ == "__main__":
    credit_df = load_credit_data()
    budget_df = load_budget_transactions()
    customer_df = load_customer_transactions()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    credit_df.to_csv(os.path.join(PROCESSED_DIR, "credit_clean.csv"), index=False)
    budget_df.to_csv(os.path.join(PROCESSED_DIR, "budget_clean.csv"), index=False)
    customer_df.to_csv(os.path.join(PROCESSED_DIR, "customer_clean.csv"), index=False)

    print(f"credit_clean: {credit_df.shape}")
    print(f"budget_clean: {budget_df.shape}")
    print(f"customer_clean: {customer_df.shape}")

    retail_df = load_retail_transactions()
    retail_df.to_csv(os.path.join(PROCESSED_DIR, "retail_clean.csv"), index=False)
    print(f"retail_clean: {retail_df.shape}")