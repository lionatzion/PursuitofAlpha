"""
model_training.py
Prepare features/labels, train a classifier, save/load models.
"""
from __future__ import annotations
from typing import Iterable, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    lookahead: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates labels: future_return > 0 over `lookahead` periods -> 1 else 0.
    Returns X, y (with final NaN lookahead rows removed).
    """
    d = df.copy()
    d["future_return"] = d["Close"].shift(-lookahead) / d["Close"] - 1.0
    d["label"] = (d["future_return"] > 0).astype(int)

    X = d[list(feature_cols)].values
    y = d["label"].values

    valid = ~np.isnan(d["future_return"].values)
    return X[valid], y[valid]


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[GradientBoostingClassifier, float]:
    """Train GradientBoosting and return (model, test_auc)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    model = GradientBoostingClassifier(max_depth=3, n_estimators=200)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    return model, auc


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
