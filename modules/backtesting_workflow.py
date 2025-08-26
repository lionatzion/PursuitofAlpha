"""
backtesting_workflow.py
End-to-end helpers for feature prep, ML strategy, and Backtrader backtesting.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import backtrader as bt

from .model_training import prepare_features  # imported for potential reuse


class MLStrategy(bt.Strategy):
    """Backtrader strategy that queries an ML model on rolling features."""
    params = dict(
        model=None,
        lookahead=3,
        buy_threshold=0.60,
        sell_threshold=0.40,
        hold_period=3,
        feature_names=("sma_10", "sma_30", "rsi_14", "z_score_50"),
    )

    def __init__(self):
        self.bars_since_entry = 0

    def next(self):
        window = np.array(self.data.close.get(size=50))
        if window.size < 50:
            return

        sma_10 = window[-10:].mean()
        sma_30 = window[-30:].mean()

        delta = np.diff(window[-15:])
        up = np.clip(delta, 0, None)
        down = -np.clip(delta, None, 0) + 1e-8
        rs = up.mean() / down.mean()
        rsi_14 = 100 - 100 / (1 + rs)

        mean50 = window[-50:].mean()
        std50 = window[-50:].std() + 1e-12
        z_50 = (window[-1] - mean50) / std50

        features = np.array([[sma_10, sma_30, rsi_14, z_50]])
        prob = float(self.p.model.predict_proba(features)[0, 1])

        if not self.position:
            if prob > self.p.buy_threshold:
                self.buy(size=1)
                self.bars_since_entry = 0
            elif prob < self.p.sell_threshold:
                self.sell(size=1)
                self.bars_since_entry = 0
        else:
            self.bars_since_entry += 1
            if self.bars_since_entry >= self.p.hold_period:
                self.close()
                self.bars_since_entry = 0


def run_backtest(model, df: pd.DataFrame, start_cash: float = 100_000.0, commission: float = 0.0005) -> float:
    """Run Backtrader backtest on a single-asset DataFrame with OHLCV columns."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by DatetimeIndex")

    data_feed = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(MLStrategy, model=model)
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=commission)

    cerebro.run()
    return float(cerebro.broker.getvalue())
