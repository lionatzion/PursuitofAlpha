import numpy as np
from sklearn.linear_model import LogisticRegression

from modules.model_evaluation import evaluate_model


def test_evaluate_model_returns_expected_metrics():
    # Simple deterministic dataset
    X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0]])
    y = (X.ravel() > 1.5).astype(int)

    clf = LogisticRegression().fit(X, y)
    metrics = evaluate_model(clf, X, y)

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    }
    assert expected_keys.issubset(metrics.keys())
    # Basic sanity checks on types and ranges
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    for k in ("tn", "fp", "fn", "tp"):
        assert isinstance(metrics[k], int)

