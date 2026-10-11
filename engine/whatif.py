"""Goal-based what-if projection.

Each month:  balance = balance * (1 + annual_return / 12) + monthly_contribution
Monthly contribution = (income - expenses - EMI) * share of surplus put towards the goal.
"""

MAX_MONTHS = 600  # stop searching after 50 years


def project_balance(current, monthly_contribution, annual_return, months):
    """Balance at the end of each month, from month 0 to `months`."""
    r = annual_return / 12
    balance = current
    balances = [balance]
    for _ in range(months):
        balance = balance * (1 + r) + monthly_contribution
        balances.append(balance)
    return balances


def months_to_goal(current, goal, monthly_contribution, annual_return):
    """Months needed to reach the goal, or None if not reached within 50 years."""
    if current >= goal:
        return 0
    r = annual_return / 12
    balance = current
    for month in range(1, MAX_MONTHS + 1):
        balance = balance * (1 + r) + monthly_contribution
        if balance >= goal:
            return month
    return None


def run_scenario(income, expenses, emi, allocation, annual_return, current, goal):
    """allocation and annual_return are fractions (0.7 = 70%, 0.06 = 6%)."""
    surplus = income - expenses - emi
    contribution = max(0.0, surplus) * allocation
    return {
        "surplus": surplus,
        "contribution": contribution,
        "months": months_to_goal(current, goal, contribution, annual_return),
    }