import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "processed", "india_households.csv")

SALARY_MIN, SALARY_MAX = 10000, 400000

# Employment mix and income spread (assumptions, stated in the report)
EMPLOYMENT = {
    "salaried":      {"p": 0.45, "median": 45000, "sigma": 0.45, "emergency_median": 3.5},
    "self_employed": {"p": 0.30, "median": 50000, "sigma": 0.70, "emergency_median": 2.5},
    "informal":      {"p": 0.25, "median": 18000, "sigma": 0.40, "emergency_median": 1.0},
}

# Loan products: (share of loans, EMI beta a, beta b, max EMI share of salary, min tenure, max tenure)
LOAN_KINDS = {
    "home":       (0.30, 3, 8, 0.50, 120, 240),
    "personal":   (0.35, 2, 8, 0.40, 12, 60),
    "vehicle":    (0.20, 2, 6, 0.30, 24, 84),
    "card_other": (0.15, 1.5, 10, 0.20, 6, 24),
}


def draw_loan_count(rng, salary):
    """Higher incomes borrow more often. About half of households have a loan overall."""
    p_any = 1 / (1 + np.exp(-(0.1 + 0.6 * np.log(salary / 35000))))
    has_loan = rng.random(len(salary)) < p_any
    extra = rng.choice([1, 2, 3], len(salary), p=[0.6, 0.3, 0.1])
    return np.where(has_loan, extra, 0)


def draw_loans(rng, salary, loan_count):
    """Up to three loans per household, each with its own product, EMI and outstanding balance."""
    n = len(salary)
    emi = np.zeros(n)
    outstanding = np.zeros(n)
    kinds = list(LOAN_KINDS)
    probs = [LOAN_KINDS[k][0] for k in kinds]

    for slot in range(3):
        active = loan_count > slot
        choice = rng.choice(len(kinds), n, p=probs)
        for idx, name in enumerate(kinds):
            _, a, b, cap, t_lo, t_hi = LOAN_KINDS[name]
            mask = active & (choice == idx)
            m = int(mask.sum())
            if m == 0:
                continue
            emi_slot = salary[mask] * rng.beta(a, b, m) * cap
            tenure = rng.integers(t_lo, t_hi + 1, m)
            remaining = tenure * rng.uniform(0.1, 1.0, m)
            emi[mask] += emi_slot
            outstanding[mask] += emi_slot * remaining * 0.85
    return emi, outstanding


def generate(n=150000, seed=42):
    rng = np.random.default_rng(seed)

    names = list(EMPLOYMENT)
    emp = rng.choice(len(names), n, p=[EMPLOYMENT[k]["p"] for k in names])
    median = np.array([EMPLOYMENT[k]["median"] for k in names])[emp]
    sigma = np.array([EMPLOYMENT[k]["sigma"] for k in names])[emp]
    salary = np.exp(np.log(median) + sigma * rng.standard_normal(n)).clip(SALARY_MIN, SALARY_MAX)
    employment = np.array(names)[emp]

    age = rng.integers(21, 60, n)
    dependents = rng.integers(0, 6, n)
    city_tier = rng.choice([1, 2, 3], n, p=[0.35, 0.35, 0.30])
    is_metro = (city_tier == 1).astype(float)

    expense_ratio = (0.60 - 0.05 * np.tanh(np.log(salary / 35000))
                     + 0.03 * dependents + 0.04 * is_metro
                     + rng.normal(0, 0.06, n)).clip(0.30, 1.20)
    expenses = salary * expense_ratio

    loan_count = draw_loan_count(rng, salary)
    emi, outstanding = draw_loans(rng, salary, loan_count)
    annual_income = salary * 12 * rng.uniform(1.0, 1.1, n)

    util = np.where(rng.random(n) < 0.6, rng.uniform(0, 30, n), rng.uniform(30, 100, n))
    emergency_median = np.array([EMPLOYMENT[k]["emergency_median"] for k in names])[emp]
    emergency = np.clip(emergency_median * np.exp(0.7 * rng.standard_normal(n)), 0, 24)

    surplus = salary - expenses - emi
    investment = np.maximum(surplus, 0) * rng.uniform(0, 0.4, n)

    # Latent risk: higher means more stretched. Ratios capture burden relative to income;
    # the surplus terms capture the rupee cushion left after expenses and EMI.
    # Noise keeps the labels imperfect.
    emi_r = emi / salary
    debt_i = outstanding / annual_income
    surplus_r = surplus / salary
    informal = (employment == "informal").astype(float)
    self_emp = (employment == "self_employed").astype(float)
    risk = (1.5 * emi_r + 0.4 * np.minimum(debt_i, 2) + 0.8 * util / 100
            - 0.15 * np.minimum(emergency, 12) / 12 - 0.5 * surplus_r
            - 0.10 * np.clip(surplus / 10000, -3, 3)
            + 0.3 * informal + 0.1 * self_emp + rng.normal(0, 0.12, n))

    q60, q85 = np.quantile(risk, [0.60, 0.85])
    health_class = np.select([risk <= q60, risk <= q85],
                             ["Stable", "Stretched"], default="Distressed")

    return pd.DataFrame({
        "monthly_salary": salary.round(0),
        "annual_income": annual_income.round(0),
        "monthly_expenses": expenses.round(0),
        "monthly_emi": emi.round(0),
        "outstanding_debt": outstanding.round(0),
        "loan_count": loan_count,
        "credit_utilization_pct": util.round(1),
        "emergency_fund_months": emergency.round(2),
        "monthly_investment": investment.round(0),
        "dependents": dependents,
        "city_tier": city_tier,
        "age": age,
        "employment_type": employment,
        "health_class": health_class,
    })


if __name__ == "__main__":
    df = generate()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} households to {OUT}\n")

    band = pd.cut(df["monthly_salary"], [10000, 20000, 50000, 100000, 200000, 400001], right=False)
    print("Salary range:")
    print(df["monthly_salary"].describe().round(0), "\n")
    print("Class mix by employment type:")
    print(pd.crosstab(df["employment_type"], df["health_class"], normalize="index").round(2), "\n")
    print("Class mix by income band:")
    print(pd.crosstab(band, df["health_class"], normalize="index").round(2), "\n")
    print("Households with a loan:", round((df["loan_count"] > 0).mean(), 3))