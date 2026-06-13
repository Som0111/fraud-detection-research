"""
fraud_utils.py
================
Shared helpers for the "When does the neural net actually win?" project.

Everything that is either (a) too long to repeat in every notebook or
(b) easy to get subtly wrong lives here. The notebooks import from this
module so the analysis cells stay short and readable.

Design choices worth knowing:
- load_transactions() prefers the REAL Kaggle file (data/creditcard.csv).
  If it is missing, it generates a synthetic stand-in with the same schema
  so the whole project still runs end to end. A banner tells you which
  path was taken.
- The dataset has no calendar dates and no cardholder id. We *simulate*
  both: time is binned into 6 "months", and cardholders are recovered by
  clustering. Both are clearly labelled as simulations, never as truth.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# Cost of each mistake, in currency units. These two numbers drive the
# entire business layer in notebook 06. A missed fraud is far more
# expensive than a false alarm, which is the whole reason the default
# 0.5 threshold is wrong.
COST_FALSE_NEGATIVE = 200.0   # fraud we let through (chargeback + loss)
COST_FALSE_POSITIVE = 5.0     # legit txn we blocked (friction, support)

N_MONTHS = 6                  # how many synthetic "months" to bin Time into
DRIFT_MONTHS = (5, 6)         # months where the new attack pattern appears
DRIFT_FEATURES = ("V14", "V4")  # PCA features whose sign we flip for drift
RANDOM_STATE = 42

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_REPO_ROOT, "data", "creditcard.csv")
ARTIFACTS_DIR = os.path.join(_REPO_ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_transactions(verbose: bool = True) -> pd.DataFrame:
    """Load the real Kaggle file if present, else a synthetic stand-in.

    Returns a DataFrame with columns: Time, V1..V28, Amount, Class.
    """
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if verbose:
            _banner(
                "REAL DATA",
                f"Loaded {len(df):,} transactions from data/creditcard.csv",
                f"Fraud rate: {df['Class'].mean():.4%}",
            )
        return df

    df = make_synthetic_transactions()
    if verbose:
        _banner(
            "SYNTHETIC DATA",
            "data/creditcard.csv not found - generated a stand-in so the",
            "pipeline runs. Drop the real file in data/ and re-run for real results.",
        )
    return df


def make_synthetic_transactions(
    n_legit: int = 60_000,
    n_fraud: int = 200,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a small dataset shaped exactly like the Kaggle file.

    Two Gaussian blobs in 28-d PCA space (legit vs fraud) plus a Time and
    Amount column. Deliberately separable enough that models can learn it,
    noisy enough that the metrics are not all 1.0. Smaller than the real
    285k rows so notebooks run fast on a laptop.
    """
    rng = np.random.default_rng(seed)
    n = n_legit + n_fraud

    # 28 PCA-like features. Fraud is shifted on a handful of dimensions,
    # mirroring how V14/V12/V10/V4 carry most of the signal in reality.
    legit = rng.normal(0.0, 1.0, size=(n_legit, 28))
    fraud = rng.normal(0.0, 1.0, size=(n_fraud, 28))
    signal_dims = [3, 9, 11, 13]   # ~ V4, V10, V12, V14 (0-indexed)
    fraud[:, signal_dims] += rng.normal(-3.0, 1.2, size=(n_fraud, len(signal_dims)))

    X = np.vstack([legit, fraud])
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)

    # Spread transactions across ~2 days of seconds, like the real Time col.
    time = rng.uniform(0, 172_792, size=n)
    # Fraud amounts skew small-but-occasionally-large; legit is broad.
    amount = np.where(
        y == 1,
        rng.gamma(1.2, 60, size=n),
        rng.gamma(2.0, 50, size=n),
    )

    cols = {"Time": time}
    for i in range(28):
        cols[f"V{i + 1}"] = X[:, i]
    cols["Amount"] = amount
    cols["Class"] = y

    df = pd.DataFrame(cols)
    # Shuffle so fraud is not all at the bottom, then re-sort by Time so the
    # temporal binning downstream is meaningful.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Temporal simulation
# --------------------------------------------------------------------------
def add_month_bins(df: pd.DataFrame, n_months: int = N_MONTHS) -> pd.DataFrame:
    """Bin the continuous Time column into N equal-width 'months'.

    This is a SIMULATION. The dataset spans ~2 days; we relabel equal time
    slices as months 1..N so we can study how models behave as time passes.
    """
    df = df.copy()
    edges = np.linspace(df["Time"].min(), df["Time"].max(), n_months + 1)
    # right=True so the max value lands in the last bin, not a new one.
    df["month"] = np.digitize(df["Time"], edges[1:-1], right=False) + 1
    df["month"] = df["month"].clip(1, n_months)
    return df


def inject_drift(
    df: pd.DataFrame,
    drift_months=DRIFT_MONTHS,
    drift_features=DRIFT_FEATURES,
    fraction: float = 0.6,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Introduce a NEW fraud pattern in the later months.

    For a fraction of fraud rows in `drift_months`, flip the sign of a few
    strongly-discriminative PCA features. A model trained only on earlier
    months has never seen this signature, so its recall should drop. This
    is the engine of the whole notebook-05 experiment.

    Adds a boolean 'is_drift' column marking the mutated rows.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    df["is_drift"] = False

    mask = (df["Class"] == 1) & (df["month"].isin(drift_months))
    fraud_idx = df.index[mask].to_numpy()
    if len(fraud_idx) == 0:
        return df

    n_pick = max(1, int(len(fraud_idx) * fraction))
    chosen = rng.choice(fraud_idx, size=n_pick, replace=False)

    for col in drift_features:
        df.loc[chosen, col] = -df.loc[chosen, col]
    df.loc[chosen, "is_drift"] = True
    return df


# --------------------------------------------------------------------------
# Synthetic cardholders (for personal-deviation features in notebook 02)
# --------------------------------------------------------------------------
def assign_synthetic_cardholders(
    df: pd.DataFrame,
    n_cards: int = 300,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Cluster transactions into pseudo-cardholders via K-Means.

    The real dataset has no user id, so we cannot compute 'how far is this
    txn from the user's normal spend'. We approximate users by clustering
    on the PCA features. Each cluster is treated as one cardholder. This is
    an approximation, labelled as such - never claim these are real users.
    """
    from sklearn.cluster import MiniBatchKMeans

    df = df.copy()
    v_cols = [c for c in df.columns if c.startswith("V")]
    km = MiniBatchKMeans(n_clusters=n_cards, random_state=seed, n_init="auto")
    df["card_id"] = km.fit_predict(df[v_cols].values)
    return df


# --------------------------------------------------------------------------
# Business cost layer
# --------------------------------------------------------------------------
def confusion_counts(y_true, y_pred):
    """Return (tn, fp, fn, tp) as plain ints."""
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def total_cost(y_true, y_pred,
               cost_fn: float = COST_FALSE_NEGATIVE,
               cost_fp: float = COST_FALSE_POSITIVE) -> float:
    """Business cost of a set of predictions: missed fraud + false alarms."""
    _, fp, fn, _ = confusion_counts(y_true, y_pred)
    return fn * cost_fn + fp * cost_fp


def cost_at_threshold(y_true, y_score, threshold: float,
                      cost_fn: float = COST_FALSE_NEGATIVE,
                      cost_fp: float = COST_FALSE_POSITIVE) -> float:
    """Total cost when we flag everything scoring >= threshold as fraud."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return total_cost(y_true, y_pred, cost_fn, cost_fp)


def sweep_thresholds(y_true, y_score, thresholds=None,
                     cost_fn: float = COST_FALSE_NEGATIVE,
                     cost_fp: float = COST_FALSE_POSITIVE) -> pd.DataFrame:
    """Evaluate cost across many thresholds. Returns a tidy DataFrame.

    Columns: threshold, tp, fp, fn, tn, precision, recall, cost.
    The row with the minimum cost is your deployment threshold.
    """
    from sklearn.metrics import precision_score, recall_score

    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        rows.append({
            "threshold": float(t),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "cost": fn * cost_fn + fp * cost_fp,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def evaluate_scores(y_true, y_score, threshold: float = 0.5) -> dict:
    """One-stop metric dict for a model's probability scores.

    Includes the metrics that matter on imbalanced data (AUC-PR, recall)
    rather than the one that lies (accuracy).
    """
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, f1_score,
        precision_score, recall_score, accuracy_score,
    )

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
 try:
    auc_roc = roc_auc_score(y_true, y_score)
except ValueError:
    auc_roc = np.nan

try:
    auc_pr = average_precision_score(y_true, y_score)
except ValueError:
    auc_pr = np.nan

return {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score(y_true, y_pred, zero_division=0),
    "f1": f1_score(y_true, y_pred, zero_division=0),
    "auc_roc": auc_roc,
    "auc_pr": auc_pr,
}


def temporal_split(df: pd.DataFrame, train_months, test_months):
    """Split by month so we never train on the future. Returns (train, test)."""
    train = df[df["month"].isin(train_months)].copy()
    test = df[df["month"].isin(test_months)].copy()
    return train, test


# --------------------------------------------------------------------------
# small internal helper
# --------------------------------------------------------------------------
def _banner(tag: str, *lines: str) -> None:
    width = max([len(tag) + 8] + [len(l) for l in lines]) + 4
    print("=" * width)
    print(f"  [{tag}]")
    for l in lines:
        print(f"  {l}")
    print("=" * width)
