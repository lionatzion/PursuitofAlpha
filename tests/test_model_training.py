import numpy as np
import pandas as pd

from modules.feature_engineering import add_indicators
from modules.model_training import prepare_features, train_classifier


def test_prepare_features_shapes_with_indicators():
    # Build a simple price series and compute indicators
    n = 150
    close = np.linspace(100, 125, n)
    raw = pd.DataFrame({"Close": close})
    feats = add_indicators(raw, sma_fast=10, sma_slow=30, rsi_window=14, z_window=50)
    feature_cols = ["sma_10", "sma_30", "rsi_14", "z_score_50"]

    X, y = prepare_features(feats, feature_cols, lookahead=3)
    # After add_indicators drops NaNs, prepare_features drops last `lookahead` rows
    assert X.shape[0] == len(feats) - 3
    assert X.shape[1] == len(feature_cols)
    assert X.shape[0] == y.shape[0]


def test_train_classifier_auc_range():
    rng = np.random.default_rng(42)
    n = 300
    # Linearly separable-ish synthetic data
    x0 = rng.normal(loc=0.0, scale=1.0, size=(n, 1))
    x1 = rng.normal(loc=0.0, scale=1.0, size=(n, 1))
    X = np.hstack([x0, x1, (x0 + x1)])
    y = ((x0 + 0.5 * x1).ravel() > 0).astype(int)

    model, auc = train_classifier(X, y, test_size=0.25, random_state=7)
    assert 0.0 <= auc <= 1.0
    # Model should implement predict_proba
    assert hasattr(model, "predict_proba")

