# FinanceTwin

A supervised model that estimates the financial health of Indian households as Stable, Stretched or Distressed, with a Streamlit dashboard for individual checks.

Built for the B.Tech (Hons) Data Science Interdisciplinary Project (DATA802).

**Live demo:** add the Streamlit link here after deployment

## What it does

- **Financial Check:** enter one household's figures, get a predicted class with probabilities, the factors driving it, suggestions, and a what-if view that re-runs the model.
- **About & Evidence:** the two-model comparison, the selection rule, the confusion matrix, performance by household type, and the limitations.

## Method

1. **Data:** 150,000 synthetic Indian households, generated from documented assumptions (`data/generate_indian_data.py`). Salaries range from ₹10,000 to ₹4,00,000 a month, with salaried, self-employed and informal workers, zero to three loans, and three city tiers. The label comes from a risk formula that combines burden relative to income with the rupee surplus left after expenses and EMI, plus noise.
2. **Features:** 14 ratios and attributes, computed by one shared function (`models/features.py`) used in both training and the app.
3. **Models:** logistic regression and random forest, compared with five-fold cross-validation on the training split. Logistic regression was selected.
4. **Evaluation:** a held-out 20% test set, used once after selection. Performance is also reported by employment type, income band, loan status and city tier.

## Results

On the held-out test set, the selected logistic regression reaches 0.877 accuracy (macro-F1 0.86), against 0.600 for a majority-class reference. Most errors fall between neighbouring classes. Accuracy is lower for informal workers (0.813) and households earning under ₹20,000 (0.826) than for salaried workers (0.903).

These results describe how well the method recovers the labelling rule on synthetic data. They do not show performance on real households.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python data/generate_indian_data.py
python models/train_model.py
python models/diagnose_groups.py
python -m streamlit run dashboard/app.py
```

The generated data and trained model are included in `data/processed/` and `models/artifacts/`, so the dashboard runs without repeating the first three steps.

## Project structure
data/
generate_indian_data.py synthetic household generator
processed/ generated dataset, and retired datasets
models/
features.py shared feature contract
train_model.py model comparison, selection and evaluation
diagnose_groups.py error analysis on the training split
artifacts/ selected model, metrics and group reports
health_score.py, forecasting.py, clustering.py exploratory, not used by the app
engine/
recommender.py exploratory, not used by the app
dashboard/
app.py Streamlit app


## Limitations

- The data is synthetic. Results show how well the method recovers its labelling rule, not how it performs on real households.
- The financial-health definition is a modelling choice, not an estimate from data.
- Errors are concentrated at the boundaries between neighbouring classes. Informal households receive more false alarms.
- The top income band has too few test households for a reliable estimate.
- Real validation needs household microdata, for example the MoSPI AIDIS 2019 unit-level data.
- This is a decision-support tool, not financial advice.