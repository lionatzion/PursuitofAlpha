import os
import numpy as np
import pandas as pd
import pytest

backtrader = pytest.importorskip("backtrader")

from modules.backtesting_workflow import run_backtest


@pytest.mark.slow
def test_backtest_runs_with_dummy_model():
    # Skip if explicitly disabled for constrained environments
    if os.getenv("BACKTEST_SKIP", "0") == "1":
        pytest.skip("Backtest smoke skipped by BACKTEST_SKIP=1")

    # Build a minimal OHLCV DataFrame (>= 50 rows for indicators used internally)
    n = 80
    idx = pd.date_range("2023-01-01", periods=n, freq="H")
    close = 100 + np.linspace(0, 2, n) + 0.5 * np.sin(np.linspace(0, 6.28, n))
    df = pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(n, 1_000_000),
        },
        index=idx,
    )

    class DummyModel:
        # Map z-score (last feature in strategy) to probability via logistic
        def predict_proba(self, X):
            # X shape: (batch, 4) -> [sma10, sma30, rsi14, z50]
            z = X[:, -1]
            p = 1.0 / (1.0 + np.exp(-z))
            return np.vstack([1 - p, p]).T

    model = DummyModel()
    final_value = run_backtest(model, df)
    assert isinstance(final_value, float)
    assert final_value > 0

