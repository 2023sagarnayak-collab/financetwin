# FinanceTwin

An intelligent personal financial planning and decision support system, built for the B.Tech Data Science Interdisciplinary Project (DATA802).

Most budgeting apps only record past spending. FinanceTwin looks forward: it scores financial health, forecasts spending, segments spending behaviour, and gives personalised suggestions.

**Live demo:**  https://financetwin-qsfufrgfvi4ctw9sy28cxs.streamlit.app/

## Features

| Module | What it does | Method |
|---|---|---|
| Financial Health Score | Scores a user 0-100 and shows how each factor contributes | Weighted score from debt-to-income, EMI burden, savings, investment and credit-use ratios |
| Spending Forecast | Projects monthly spending | Exponential smoothing, backtested against baselines |
| Customer Segments | Groups customers into active and lapsed spenders | RFM features, log scaling, k-means |
| Recommendations | Suggests what to improve | Transparent rule-based thresholds on raw ratios |
| What-If Simulator | Shows how long a savings goal takes under different income, spending and saving choices | Month-by-month compound-growth projection |

## Datasets (all public, from Kaggle)

- `parisrohan/credit-score-classification` for the Health Score
- `ismetsemedov/personal-budget-transactions-dataset` for forecasting
- `mashlyn/online-retail-ii-uci` for customer segments
- `bkcoban/customer-transactions` was explored for segmentation, then replaced because each customer had only one transaction

## Key findings

- **Health Score:** average scores follow the real credit labels (Good about 82, Standard about 79, Poor about 75). Debt-to-income and EMI burden separated the groups best, so they carry 80% of the weight.
- **Forecasting:** no method beat the naive "repeat last month" baseline, because individual monthly spending is noisy.
- **Segmentation:** log-scaling RFM features removed the distortion from a few extreme wholesale buyers and gave two interpretable segments.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

The processed data needed by the dashboard is included in `data/processed/`.

### Rebuild the data from scratch (optional)

1. Download the datasets into `data/raw/` with the Kaggle CLI, for example `kaggle datasets download -d <slug> -p data/raw --unzip`.
2. Rename `11 march 2025.csv` to `budget_transactions.csv` and `sample_dataset.csv` to `customer_transactions.csv`.
3. Run `python data/data_loader.py`, then `python models/clustering.py`.

## Project structure

```
data/         data_loader.py and processed datasets
models/       health_score.py, forecasting.py, clustering.py
engine/       recommender.py
dashboard/    app.py (Streamlit app)
notebooks/    eda.ipynb
```

## Limitations

- Datasets are public and partly synthetic, so results show the method works rather than describing real people.
- About 18% of credit records were dropped because of missing salary values.
- Weights and thresholds are design choices calibrated to these datasets and would need recalibration for region-specific data.
- Amounts on the Forecast and Segments pages are in the datasets' own units. The Health Score uses ratios, so it does not depend on currency.
- This is a decision-support tool, not financial advice.
