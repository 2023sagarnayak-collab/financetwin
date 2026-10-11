import pandas as pd
import numpy as np
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

MIN_TRANSACTIONS_TO_FORECAST = 200  # categories below this are too sparse


def load_monthly_series(category: str = None) -> pd.Series:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "budget_clean.csv"))
    df["date"] = pd.to_datetime(df["date"])

    if category:
        df = df[df["category"] == category]

    monthly = df.set_index("date")["amount"].resample("MS").sum()
    monthly = monthly[:-1]  # drop the last month — likely partial/incomplete
    return monthly


def backtest_forecast(series: pd.Series, test_months: int = 6):
    train, test = series[:-test_months], series[-test_months:]
    naive_baseline = pd.Series([train.iloc[-1]] * test_months, index=test.index)
    naive_rmse = np.sqrt(((naive_baseline - test) ** 2).mean())

    results = {}

    # Trend-based model (what we tried first)
    model_trend = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
    forecast_trend = model_trend.forecast(test_months)
    results["trend"] = (forecast_trend, np.sqrt(((forecast_trend - test) ** 2).mean()))

    # No-trend model — just smooths the level, doesn't extrapolate direction
    model_no_trend = ExponentialSmoothing(train, trend=None, seasonal=None).fit()
    forecast_no_trend = model_no_trend.forecast(test_months)
    results["no_trend"] = (forecast_no_trend, np.sqrt(((forecast_no_trend - test) ** 2).mean()))

    # Simple 3-month moving average
    ma_value = train[-3:].mean()
    forecast_ma = pd.Series([ma_value] * test_months, index=test.index)
    results["moving_avg_3m"] = (forecast_ma, np.sqrt(((forecast_ma - test) ** 2).mean()))

    print(f"Naive baseline RMSE: {naive_rmse:.2f}")
    for name, (fc, rmse) in results.items():
        beats = "beats" if rmse < naive_rmse else "does NOT beat"
        print(f"{name} RMSE: {rmse:.2f} ({beats} naive)")

    best_name = min(results, key=lambda k: results[k][1])
    print(f"\nBest model: {best_name}")

    return results, naive_baseline, naive_rmse


def forecast_future(series: pd.Series, periods: int = 3) -> pd.Series:
    """Forecast the next N months using the full available history.
    Uses no-trend smoothing — backtesting showed trend-based models
    overreact to one-off spending spikes with this much data."""
    model = ExponentialSmoothing(series, trend=None, seasonal=None).fit()
    return model.forecast(periods)

if __name__ == "__main__":
    total_series = load_monthly_series()
    print(f"Monthly series length: {len(total_series)} months")
    print(total_series)

    results, naive_baseline, naive_rmse = backtest_forecast(total_series)

    future = forecast_future(total_series, periods=3)
    print(f"\nNext 3 months forecast:\n{future}")