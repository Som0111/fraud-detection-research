"""
models.py
=========
Model factories shared by notebook 03 (standard comparison) and notebook 05
(temporal drift). Defining the models in one place guarantees that the
"same" model really is the same in both experiments - otherwise a difference
in results could just be a difference in hyper-parameters.
"""

from __future__ import annotations
import numpy as np

RANDOM_STATE = 42


def get_sklearn_models(scale_pos_weight: float) -> dict:
    """Return the four non-neural models, configured for imbalance.

    All of them get class-weighting so the rare fraud class is not ignored.
    scale_pos_weight = (#negatives / #positives) computed on the TRAIN set.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        "logreg": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
        ),
        "tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }


def build_keras_mlp(input_dim: int, l2: float = 0.0, dropout: float = 0.0):
    """A small dense network: the architecture from the project plan.

        Input -> Dense(64, ReLU) -> [Dropout] -> Dense(32, ReLU) ->
                 [Dropout] -> Dense(1, Sigmoid)

    Pass l2=0, dropout=0 to get the UNREGULARISED version (which will
    overfit), or l2>0, dropout>0 for the regularised one. Loss is binary
    cross-entropy, the natural choice for a 0/1 target.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers, models

    tf.random.set_seed(RANDOM_STATE)
    reg = regularizers.l2(l2) if l2 > 0 else None

    net = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dropout(dropout) if dropout > 0 else layers.Activation("linear"),
        layers.Dense(32, activation="relu", kernel_regularizer=reg),
        layers.Dropout(dropout) if dropout > 0 else layers.Activation("linear"),
        layers.Dense(1, activation="sigmoid"),
    ])
    net.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc_pr", curve="PR")],
    )
    return net


def class_weight_dict(y) -> dict:
    """Keras-style class weights: inverse frequency, normalised."""
    y = np.asarray(y)
    n = len(y)
    n_pos = max(int(y.sum()), 1)
    n_neg = n - n_pos
    return {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}


def pos_weight(y) -> float:
    """scale_pos_weight for XGBoost: negatives / positives on the train set."""
    y = np.asarray(y)
    n_pos = max(int(y.sum()), 1)
    return float((len(y) - n_pos) / n_pos)
