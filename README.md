# FraudLens: When Does a Neural Network Actually Win?

A fraud-detection project comparing traditional machine learning and neural networks under real-world conditions.

Most fraud-detection projects stop after training a classifier and reporting accuracy. This project goes further by asking a more practical question:

> If a fraud pattern changes over time, which model holds up best, and how often do you need to retrain it?

To explore this, I built an end-to-end fraud-detection pipeline using the Credit Card Fraud Detection dataset and introduced a simulated temporal drift scenario where a new fraud pattern appears later in the data. The goal was not just to compare models, but to understand how they behave when the environment changes.

---

## What this project covers

| Topic                                                      | Notebook |
| ---------------------------------------------------------- | -------- |
| Exploratory Data Analysis (EDA) and class imbalance        | NB-01    |
| Feature engineering                                        | NB-02    |
| Logistic Regression, Decision Tree, Random Forest, XGBoost | NB-03    |
| Neural Networks, Regularisation, Overfitting               | NB-03    |
| AUC-PR, ROC, Calibration, Confusion Matrices               | NB-04    |
| Temporal Drift and Model Robustness                        | NB-05    |
| Threshold Optimisation and Business Costs                  | NB-06    |

---

## Project Workflow

### 01. Data Exploration

* Explore the severe fraud imbalance
* Create six synthetic time periods ("months")
* Inject a new fraud pattern into the final months
* Verify the distribution shift using KS tests

### 02. Feature Engineering

Built additional fraud-oriented features such as:

* Transaction velocity
* Personal spending deviation (z-score)
* Time-of-day cyclic features
* Amount transformations

### 03. Model Arena

Trained and compared:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Neural Network (with and without regularisation)

The neural network is intentionally trained twice so the effect of overfitting and regularisation can be visualised directly.

### 04. Evaluation Lab

This notebook demonstrates why accuracy is a poor metric for fraud detection.

Models are compared using:

* Precision-Recall Curves
* ROC Curves
* AUC-PR
* Calibration Curves
* Confusion Matrices

### 05. Temporal Drift Experiment

This is the core experiment.

Models are trained on Months 1–4 and evaluated on Months 5–6 after the new fraud pattern appears.

The notebook measures:

* Performance degradation under drift
* Recall on the new fraud pattern specifically
* Recovery after retraining

### 06. Business Decision Engine

The final notebook converts model outputs into business decisions.

It includes:

* Cost-sensitive threshold optimisation
* Expected-loss analysis
* Precision-recall trade-offs
* A final deployment recommendation

---

## Dataset

This project uses the Kaggle Credit Card Fraud Detection dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in numerical order.

---

## Project Structure

```text
fraud-detection-research/
├── README.md
├── requirements.txt
├── data/
├── src/
│   ├── fraud_utils.py
│   └── models.py
├── notebooks/
│   ├── 01_data_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_arena.ipynb
│   ├── 04_evaluation_lab.ipynb
│   ├── 05_temporal_drift.ipynb
│   └── 06_business_decision_engine.ipynb
└── artifacts/
```

---

## Key Findings

* Accuracy is misleading for highly imbalanced fraud datasets; AUC-PR provides a much more useful comparison.
* XGBoost delivered the strongest overall performance on static tabular data.
* Every model degraded when the fraud pattern changed over time.
* Retraining frequency mattered as much as model choice under temporal drift.
* The optimal deployment threshold was far from the default 0.50 because the cost of missed fraud was much higher than the cost of false alarms.
* Effective fraud detection requires both a good model and a business-aware decision threshold.

---

## Final Takeaway

The most interesting result was that model selection alone did not determine success. Under temporal drift, performance depended heavily on how quickly the model was retrained after new fraud behaviour appeared.

In other words, the real deployment decision is not just **which model to use**, but also **how often to update it**.
