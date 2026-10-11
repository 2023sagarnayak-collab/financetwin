# Each rule checks a raw ratio against a threshold.
# direction: "above" = flag when value is higher than threshold (bad if high)
#            "below" = flag when value is lower than threshold (bad if low)
# Thresholds are tunable design choices:
#   debt_to_income 0.10  ~ average of the "Poor" credit group in our validation (0.099)
#   emi_burden     0.60  ~ between the "Standard" (0.53) and "Poor" (0.68) group averages
#   savings_rate   0.20  ~ the commonly cited "save 20%" guideline (50/30/20 rule)
#   credit_util    0.30  ~ the commonly cited "stay under 30%" guideline
#   investment     0.10  ~ a commonly cited starting guideline of ~10% of income
RULES = [
    {
        "metric": "debt_to_income",
        "direction": "above",
        "threshold": 0.10,
        "title": "High debt relative to income",
        "message": "Your outstanding debt is {value:.0%} of your annual income. Consider "
                   "prioritizing repayment of high-interest debt before increasing "
                   "discretionary spending.",
        "priority": 1,
    },
    {
        "metric": "emi_burden",
        "direction": "above",
        "threshold": 0.40,
        "title": "Heavy monthly loan repayment burden",
        "message": "Loan EMIs take up {value:.0%} of your monthly salary. If possible, explore "
                   "refinancing or consolidating loans to reduce this burden.",
        "priority": 2,
    },
    {
        "metric": "savings_rate",
        "direction": "below",
        "threshold": 0.20,
        "title": "Low savings relative to income",
        "message": "You're keeping {value:.0%} of your monthly salary as balance. Aiming for "
                   "around 20% (for example via an automatic transfer on payday) builds a "
                   "meaningful buffer over time.",
        "priority": 3,
    },
    {
        "metric": "credit_utilization",
        "direction": "above",
        "threshold": 0.30,
        "title": "High credit utilization",
        "message": "You're using {value:.0%} of your available credit. Keeping utilization "
                   "under about 30% generally supports better credit health.",
        "priority": 4,
    },
    {
        "metric": "investment_ratio",
        "direction": "below",
        "threshold": 0.10,
        "title": "Limited investment activity",
        "message": "You're investing {value:.0%} of your monthly salary. Once your emergency "
                   "fund and debt are under control, consider a small, consistent monthly "
                   "investment contribution.",
        "priority": 5,
    },
]


def generate_recommendations(metrics: dict) -> list[dict]:
    """metrics: dict of raw ratios keyed by RULES[i]['metric']. Returns triggered
    recommendations, most urgent first, with the user's actual value filled in."""
    triggered = []
    for rule in RULES:
        value = metrics.get(rule["metric"])
        if value is None:
            continue
        hit = value > rule["threshold"] if rule["direction"] == "above" else value < rule["threshold"]
        if hit:
            triggered.append({
                "title": rule["title"],
                "message": rule["message"].format(value=value),
                "priority": rule["priority"],
            })

    triggered.sort(key=lambda r: r["priority"])

    if not triggered:
        return [{
            "title": "No major issues detected",
            "message": "Your financial indicators look healthy across the board. "
                       "Keep monitoring your Health Score as your situation changes.",
        }]
    return triggered