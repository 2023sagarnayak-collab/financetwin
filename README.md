# FinanceTwin

FinanceTwin is a money-health checker for Indian households. Enter your pay, spending, loan payments and savings, and it tells you whether your finances are **On track**, **Getting tight** or **Under strain**. It then shows what is behind the result, suggests the simplest changes that could move you to On track, and estimates how long it takes to build a safety cushion.

Built for the B.Tech (Hons) Data Science Interdisciplinary Project (DATA802).

**Live demo:** add the Streamlit link here after deployment

> **Please note:** the model was trained and tested on synthetic (made-up) households. It shows how the method works, not how it performs on real people. It is not financial advice.

## What the app does

- **Home:** what the tool does and how to use it, with an accuracy note.
- **Check my money health:** enter your figures and get a result, the main factors behind it, and suggestions. A "Try a change" section re-scores your household after changes to pay, spending, loan payments or savings. An action plan suggests the simplest combination of changes that could move you to On track, and a timeline shows how long a safety cushion would take.
- **How it works:** a plain-language explanation, a glossary, and the technical results (model comparison, confusion matrix and accuracy by household type) in an expandable section.

## Method

1. **Data:** 150,000 synthetic Indian households, generated from documented assumptions (`data/generate_indian_data.py`). Pay ranges from ₹10,000 to ₹4,00,000 a month, with salaried, self-employed and informal workers, zero to three loans, and three city tiers. City tier affects living costs, which is the only route by which it matters.
2. **Financial-health label:** a risk score that combines burden relative to income (loan payments and debt as shares of pay or income) with the rupee amount left at the end of the month, plus noise. Both parts are our assumptions about what financial health means, not estimates from data.
3. **Features:** 14 ratios and attributes, computed by one shared function (`models/features.py`) used in both training and the app.
4. **Models:** logistic regression and random forest, compared with five-fold cross-validation on the training split. Logistic regression was selected. The held-out 20% test set was used once, after selection.
5. **Evaluation:** accuracy and macro-F1 overall, and by employment type, income band, loan status and city tier, plus an error analysis on the training split.

## Results

On the held-out test set, the selected logistic regression reaches 0.877 accuracy (macro-F1 0.86), against 0.600 for a majority-class reference. Most mistakes are between neighbouring classes, such as On track and Getting tight. Accuracy is lower for informal workers (0.813) and households earning under ₹20,000 (0.826) than for salaried workers (0.903). About 24% of Stable informal households are flagged as Stretched, against about 8% of salaried households.

### Trade-off in the income definition

We tested three definitions of financial health:

| Definition | Overall test accuracy | Informal workers | Earning under ₹20,000 |
|---|---|---|---|
| Ratios only | 0.871 | 0.826 | 0.844 |
| Ratios plus a low-salary buffer | 0.869 | 0.797 | 0.810 |
| Ratios plus rupee surplus (used) | 0.877 | 0.813 | 0.826 |

The rupee-surplus definition improves overall accuracy but costs some accuracy for informal and low-income households compared with ratios alone. We report that trade-off rather than choosing the setting that looks best overall.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

The generated dataset and trained model are included in `data/processed/` and `models/artifacts/`, so the app runs without the training steps. To rebuild everything from scratch, run these in order:

```bash
python data/generate_indian_data.py
python models/train_model.py
python models/diagnose_groups.py
python models/segment_households.py
```

## Project structure

```
dashboard/
  app.py                       Streamlit app (Home, Check my money health, How it works)
data/
  generate_indian_data.py      synthetic household generator and label definition
  processed/                   generated dataset (india_households.csv) and retired datasets
models/
  features.py                  shared feature contract used by training and the app
  train_model.py               two-model comparison, selection and evaluation
  diagnose_groups.py           error analysis on the training split
  segment_households.py        exploratory household segments (not shown in the app)
  artifacts/                   selected model, metrics, group reports, segment outputs
  health_score.py, forecasting.py, clustering.py   earlier exploratory work (not used by the app)
engine/
  recommender.py               earlier exploratory work (not used by the app)
```

## Limitations

- **Synthetic data.** High accuracy shows the model recovers its labelling rule, not that it predicts real households. Real validation needs household microdata, for example the MoSPI AIDIS 2019 unit-level data.
- **The definition of financial health is a choice.** The model learns the rule we wrote. Real outcomes would show whether the definition predicts actual hardship.
- **Location is indirect.** City tier affects the model only through living costs. Rent is not measured separately, so the model cannot see location's effect on housing costs directly.
- **Informal and low-income households** receive more false alarms, and their accuracy is lower than for salaried households.
- **Boundary errors.** Most mistakes fall between neighbouring classes. Mistakes straight from On track to Under strain are rare.
- **Thin top income band.** The ₹2L to ₹4L band has few test households, so its results are unreliable.
- **Action plan is a rule of thumb.** It searches a fixed grid of changes, and its "simplest" ranking uses assumed costs. Refinancing is assumed to lower monthly payments without changing the amount owed.
- **Not financial advice.** The tool estimates, it does not advise. For decisions about loans, investments or debt, speak to a qualified adviser.