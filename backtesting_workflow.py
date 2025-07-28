"""
backtesting_workflow.py

This module provides a sketch of a workflow for downloading market data,
engineering features, training a classification model and running a backtest
using Backtrader.  It is designed to be imported into a Google Colab
notebook or executed as a standalone script.  It illustrates how to
structure code for testing quantitative trading strategies with Python.

Dependencies:
  pandas, numpy, yfinance, scikit‑learn, backtrader and ta

Note: This code is a simplified example.  In practice you should
implement more robust data handling, cross‑validation (e.g. purged
cross‑validation), proper position sizing and risk management.
"""

# ---------------------------------------------------------------------------
# Runtime dependency installation
#
# The following block attempts to import the third‑party packages used by
# this module.  If an ImportError is raised because the package is not
# installed in the execution environment, it will automatically install
# the missing package using pip.  This ensures that users running this
# script in a fresh Google Colab or other environment do not encounter
# `ModuleNotFoundError` for `backtrader`, `yfinance` or `ta`.
#
# Installing packages at import time will slow down the initial load of
# this module and requires network access.  If these packages are already
# installed, the installation step is skipped.

import subprocess
import sys

def _install_and_import(pkg_name: str) -> None:
    """Install and import a package if it is missing.

    Parameters
    ----------
    pkg_name : str
        Name of the package to ensure is available.

    Side Effects
    ------------
    If the package is not available, this function will run a pip
    installation command using the current Python executable and then
    attempt to import the package again.  This can take time and
    requires an internet connection.
    """
    try:
        __import__(pkg_name)
    except ImportError:
        # Attempt to install the missing package
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        # Retry the import after installation
        __import__(pkg_name)

# Ensure essential third‑party libraries are available
for _package in ("backtrader", "yfinance", "ta"):
    _install_and_import(_package)

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from ta import trend, volatility

# Machine learning
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Backtesting
import backtrader as bt


def download_data(tickers, start, end, interval="1h"):
    """Download OHLCV data for a list of tickers using yfinance.

    Parameters
    ----------
    tickers : list[str]
        Symbols to download.
    start : str or datetime
        Start date in 'YYYY-MM-DD' format or datetime.
    end : str or datetime
        End date in 'YYYY-MM-DD' format or datetime.
    interval : str, optional
        Data frequency (e.g. '1h', '30m', '1d').  Default is '1h'.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping of ticker to DataFrame containing OHLCV data.
    """
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if not df.empty:
            df.dropna(inplace=True)
            data[ticker] = df
    return data


def create_volume_bars(df, vol_target=1e6):
    """Aggregate tick/interval data into volume bars.

    This function groups rows until the cumulative traded volume reaches
    `vol_target` and then starts a new bar.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with at least 'Open', 'High', 'Low', 'Close', 'Volume' columns.
    vol_target : float
        Total volume per bar.

    Returns
    -------
    pandas.DataFrame
        Resampled data using volume bars.
    """
    bars = []
    cum_vol = 0
    bar_rows = []
    for idx, row in df.iterrows():
        bar_rows.append(row)
        cum_vol += row["Volume"]
        if cum_vol >= vol_target:
            bar_df = pd.DataFrame(bar_rows)
            # Create OHLCV bar
            o = bar_df.iloc[0]["Open"]
            h = bar_df["High"].max()
            l = bar_df["Low"].min()
            c = bar_df.iloc[-1]["Close"]
            v = bar_df["Volume"].sum()
            bars.append({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
            # reset
            bar_rows = []
            cum_vol = 0
    return pd.DataFrame(bars)


def add_indicators(df, sma_fast=10, sma_slow=30, rsi_window=14, z_window=50):
    """Add technical indicators (SMA, RSI, Z-score) to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Close' column.
    sma_fast : int
        Window for the fast simple moving average.
    sma_slow : int
        Window for the slow simple moving average.
    rsi_window : int
        Window for Relative Strength Index.
    z_window : int
        Window for Z-score (mean and std calculation).

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional indicator columns.
    """
    df = df.copy()
    df[f"sma_{sma_fast}"] = df["Close"].rolling(window=sma_fast).mean()
    df[f"sma_{sma_slow}"] = df["Close"].rolling(window=sma_slow).mean()
    df[f"rsi_{rsi_window}"] = volatility.rsi(df["Close"], window=rsi_window)
    df[f"z_score_{z_window}"] = (df["Close"] - df["Close"].rolling(window=z_window).mean()) / df["Close"].rolling(window=z_window).std()
    df.dropna(inplace=True)
    return df


def prepare_features(df, feature_cols, lookahead=3):
    """Prepare feature matrix and labels for classification.

    Labels are defined as 1 if the future return over `lookahead` periods is >0,
    otherwise 0.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing indicator columns.
    feature_cols : list[str]
        List of column names to use as features.
    lookahead : int
        Number of periods ahead to compute future return.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Feature matrix X and labels y.
    """
    df = df.copy()
    df["future_return"] = df["Close"].shift(-lookahead) / df["Close"] - 1
    df["label"] = (df["future_return"] > 0).astype(int)
    X = df[feature_cols].values
    y = df["label"].values
    # Drop the last lookahead rows where future_return is NaN
    valid_rows = ~np.isnan(df["future_return"]).values
    return X[valid_rows], y[valid_rows]


def train_classifier(X, y, test_size=0.2, random_state=42):
    """Train a Gradient Boosting classifier and return the trained model.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Binary labels.
    test_size : float
        Fraction of data to hold out as test.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn.ensemble.GradientBoostingClassifier
        Trained model.
    float
        ROC AUC score on held‑out test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    model = GradientBoostingClassifier(max_depth=3, n_estimators=200)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return model, auc


class MLStrategy(bt.Strategy):
    """Backtrader strategy that uses an ML model to generate trading signals.

    Buy when model probability > buy_threshold, sell/short when probability < sell_threshold.
    Close positions after a fixed hold period.
    """

    params = dict(
        model=None,  # Trained classifier with predict_proba
        feature_cols=None,  # List of columns used to compute features
        lookahead=3,
        buy_threshold=0.6,
        sell_threshold=0.4,
        hold_period=3,
    )

    def __init__(self):
        # Reference to close price for convenience
        self.dataclose = self.datas[0].close
        self.bars_since_entry = 0

    def next(self):
        # Compute indicators on the fly for the most recent bar
        # Note: In Backtrader, we typically precompute indicators.
        # Here we extract values from the strategy's data feed.
        window_data = self.data.close.get(size=max(self.params.lookahead, 50))
        # Only proceed if enough data points are available
        if len(window_data) < max(self.params.lookahead, 50):
            return
        # Compute features similar to training
        sma_fast = np.mean(window_data[-10:])
        sma_slow = np.mean(window_data[-30:])
        # RSI calculation (simplified)
        delta = np.diff(window_data[-15:])
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_gain = up.mean()
        avg_loss = abs(down.mean()) + 1e-8  # avoid division by zero
        rs = avg_gain / avg_loss
        rsi_val = 100 - 100 / (1 + rs)
        mean = np.mean(window_data[-50:])
        std = np.std(window_data[-50:])
        z_score = (window_data[-1] - mean) / std
        features = np.array([[sma_fast, sma_slow, rsi_val, z_score]])
        prob = self.params.model.predict_proba(features)[0, 1]
        # Entry signals
        if not self.position:
            if prob > self.params.buy_threshold:
                self.buy(size=1)
                self.bars_since_entry = 0
            elif prob < self.params.sell_threshold:
                self.sell(size=1)
                self.bars_since_entry = 0
        # Exit condition
        else:
            self.bars_since_entry += 1
            if self.bars_since_entry >= self.params.hold_period:
                self.close()
                self.bars_since_entry = 0


def run_backtest(model, data_df, start_cash=100000.0, commission=0.0005):
    """Run a backtest using Backtrader and the provided model.

    Parameters
    ----------
    model : classifier
        Trained model that supports predict_proba.
    data_df : pandas.DataFrame
        DataFrame with OHLCV data and indicator columns.
    start_cash : float
        Initial capital for the backtest.
    commission : float
        Commission rate per trade (e.g. 0.0005 for 0.05%).

    Returns
    -------
    float
        Final portfolio value.
    """
    cerebro = bt.Cerebro()
    # Convert DataFrame to Backtrader data feed
    data_feed = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data_feed)
    # Add strategy
    cerebro.addstrategy(
        MLStrategy,
        model=model,
        feature_cols=["sma_10", "sma_30", "rsi_14", "z_score_50"],
    )
    # Set initial cash and commission
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=commission)
    # Run the backtest
    cerebro.run()
    final_value = cerebro.broker.getvalue()
    return final_value


if __name__ == "__main__":
    # Example usage: download data, prepare features, train model and backtest
    tickers = ["AAPL"]
    data = download_data(tickers, start="2019-01-01", end="2024-12-31", interval="1h")
    if not data:
        raise SystemExit("No data downloaded.")
    # Process the first ticker for demonstration
    df = create_volume_bars(data[tickers[0]], vol_target=5e6)
    df = add_indicators(df, sma_fast=10, sma_slow=30, rsi_window=14, z_window=50)
    feature_cols = ["sma_10", "sma_30", "rsi_14", "z_score_50"]
    X, y = prepare_features(df, feature_cols)
    model, auc = train_classifier(X, y)
    print(f"Trained model ROC AUC: {auc:.4f}")
    final = run_backtest(model, df)
    print(f"Final portfolio value: {final:.2f}")
