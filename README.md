# When does the neural net actually win?

### An advanced credit-card fraud detection study

This is not a generic "train a classifier on `creditcard.csv`" project. It is a
small piece of **research** that answers a real question:

> Neural networks are expressive — but are they actually *better* than gradient
> boosting on tabular fraud data, and what breaks that advantage in production?

The twist that sets it apart: we simulate **temporal drift** — a new fraud
pattern emerging mid-dataset — and measure which models adapt, which collapse,
and which recover. That is what actually breaks deployed fraud systems, and
almost no student project touches it.

---

## What it covers

Everything from a foundational ML + deep-learning course, used in anger:

| Area | Where |
|---|---|
| Logistic regression, decision tree, random forest, XGBoost | NB-03 |
| Dense layers, ReLU, sigmoid, BCE loss, L2, dropout, overfitting | NB-03 |
| EDA, class imbalance, distribution shift (KS test) | NB-01 |
| Feature engineering (velocity, personal deviation, cyclic time) | NB-02 |
| Metrics: AUC-PR, ROC, F1, calibration, confusion matrices | NB-04 |
| Temporal drift, robustness, retraining cadence | NB-05 |
| Cost analysis, threshold selection, business recommendation | NB-06 |

---

## The six notebooks

Run them **in order** — each saves an artifact the next one loads.

1. **`01_data_and_eda.ipynb`** — load data, expose the imbalance, bin time into
   six "months", and inject a new fraud pattern into the last two. KS test
   confirms the drift is real.
2. **`02_feature_engineering.ipynb`** — synthetic cardholders (via clustering),
   transaction velocity, personal-deviation z-score, cyclic time-of-day, amount
   transforms.
3. **`03_model_arena.ipynb`** — five models on one stratified split. The neural
   net is trained twice (with and without dropout + L2) so you can *see*
   overfitting and its fix in the loss curves.
4. **`04_evaluation_lab.ipynb`** — rank by accuracy (everything looks great),
   then switch to AUC-PR and watch the ranking flip. PR / ROC / calibration
   curves and confusion matrices.
5. **`05_temporal_drift.ipynb`** — the centerpiece. Train on months 1–4, test on
   5–6. Measure degradation, isolate recall on the *new* pattern, then retrain
   and watch the neural net recover.
6. **`06_business_decision_engine.ipynb`** — turn probabilities into money. Cost
   matrix, threshold sweep, expected-loss curve, and one actionable sentence.

---

## Setup

```bash
pip install -r requirements.txt
```

Then launch Jupyter and run the notebooks in order:

```bash
jupyter lab        # or jupyter notebook
```

## The dataset

The project is built around the **Kaggle Credit Card Fraud Detection** dataset
(ULB Machine Learning Group):

- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- File: `creditcard.csv` (~144 MB, 284,807 rows; columns `Time`, `V1`–`V28`,
  `Amount`, `Class`).

**Drop `creditcard.csv` into the `data/` folder, then run every notebook in
order.**

### Runs out of the box without the dataset

If `data/creditcard.csv` is missing, the code automatically generates a small
**synthetic** dataset with the identical schema, so the whole pipeline runs end
to end immediately. A banner in NB-01 tells you which path was taken. **The
outputs currently saved in the notebooks were produced on synthetic data** —
your numbers will differ (and be more interesting) on the real file.

---

## Project layout

```
fraud-detection-research/
├── README.md
├── requirements.txt
├── data/
│   └── creditcard.csv        <- you add this (Kaggle)
├── src/
│   ├── fraud_utils.py        <- loading, drift injection, cost matrix, metrics
│   └── models.py             <- model factories shared by NB-03 and NB-05
├── notebooks/
│   ├── 01_data_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_arena.ipynb
│   ├── 04_evaluation_lab.ipynb
│   ├── 05_temporal_drift.ipynb
│   └── 06_business_decision_engine.ipynb
└── artifacts/                <- intermediate parquet files land here at runtime
```

---

## What the project concludes

1. **Accuracy is the wrong metric** for fraud — every model clears 99%. AUC-PR is
   the honest ranking.
2. On **static tabular data**, gradient boosting (XGBoost) is hard to beat; the
   neural net only matches it with careful regularisation.
3. Under **temporal drift**, every model degrades — and the neural net degrades
   hardest, then recovers fastest when retrained.
4. The deployment decision is **economic, not statistical**: choose the model,
   the threshold, *and* the retraining cadence together.

So "is the neural net better?" has no unconditional answer. It depends on your
retraining cadence — which is a real production architecture decision, backed
here by your own experiment.
