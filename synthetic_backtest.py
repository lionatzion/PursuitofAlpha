"""
Synthetic Backtesting Prototype
===============================

This script generates a synthetic price series and demonstrates a simple
workflow for feature engineering, model training and backtesting using
only the libraries that are available in this environment.  External
data sources are not used because internet access is restricted in
the running container.  Instead, a random walk with a small positive
drift is used to approximate the behaviour of a financial time series.

Key steps:

1. **Generate Synthetic Prices** – A price series is simulated using
   geometric Brownian motion with a configurable number of steps.  The
   resulting series mimics daily closing prices for a single asset.

2. **Feature Engineering** – Common technical indicators such as
   simple moving averages (10‑day and 30‑day), a 14‑day RSI and a
   50‑day rolling Z‑score are computed.  These features are used as
   inputs to the classification model.

3. **Label Construction** – A binary label is assigned based on
   whether the return over the next `lookahead` periods is positive
   or negative.

4. **Model Training** – A gradient boosting classifier from
   CatBoost is trained on a portion of the data.  CatBoost is chosen
   because it is available in this environment and provides solid
   performance out‑of‑the‑box.  The AUC on a held‑out test set is
   reported.

5. **Backtesting** – A simple rule‑based strategy is run over the
   data.  When the model’s predicted probability exceeds the
   `buy_threshold` the strategy goes long one unit of the asset;
   when it falls below the `sell_threshold` it goes short.  Trades
   are held for a fixed number of periods before closing.  The final
   portfolio value is reported.

This prototype is intended as a proof of concept for a larger
quantitative research pipeline.  It illustrates how to structure
code for simulation and analysis without relying on live market data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

try:
    # CatBoost is used for the classification model.  It is available
    # in this environment by default.  If it cannot be imported
    # something is wrong with the installation.
    from catboost import CatBoostClassifier
except ImportError as e:
    raise ImportError(
        "CatBoost library is required for this script but could not be imported."
    ) from e


@dataclass
class BacktestParameters:
    """Configuration parameters for the synthetic backtest."""

    n_steps: int = 1000  # number of time steps in the synthetic series
    drift: float = 0.0002  # daily drift in log‑returns
    volatility: float = 0.01  # daily volatility in log‑returns
    seed: int = 42  # random seed for reproducibility
    lookahead: int = 3  # number of days ahead to compute the target
    buy_threshold: float = 0.6  # probability threshold to enter long
    sell_threshold: float = 0.4  # probability threshold to enter short
    hold_period: int = 3  # number of periods to hold a position


def generate_price_series(params: BacktestParameters) -> pd.DataFrame:
    """Generate a synthetic price series using geometric Brownian motion.

    Parameters
    ----------
    params : BacktestParameters
        Configuration with drift, volatility and number of steps.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the synthetic 'Close' prices indexed by time.
    """
    np.random.seed(params.seed)
    # Generate log returns
    log_returns = params.drift + params.volatility * np.random.randn(params.n_steps)
    log_price = np.cumsum(log_returns)
    prices = np.exp(log_price)
    # Normalise the series to start at 100
    prices = prices / prices[0] * 100.0
    df = pd.DataFrame({"Close": prices})
    df.index.name = "Time"
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicator features.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'Close' column.

    Returns
    -------
    pandas.DataFrame
        DataFrame including the computed features.
    """
    result = df.copy()
    # Simple moving averages
    result["sma_10"] = result["Close"].rolling(window=10).mean()
    result["sma_30"] = result["Close"].rolling(window=30).mean()
    # RSI calculation
    delta = result["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use exponential moving average for smoothing
    roll_up = gain.ewm(span=14, adjust=False).mean()
    roll_down = loss.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-8)
    result["rsi_14"] = 100 - 100 / (1 + rs)
    # Z-score (rolling)
    roll_mean = result["Close"].rolling(window=50).mean()
    roll_std = result["Close"].rolling(window=50).std()
    result["z_score_50"] = (result["Close"] - roll_mean) / roll_std
    result.dropna(inplace=True)
    return result


def prepare_dataset(
    df: pd.DataFrame, feature_cols: Tuple[str, ...], lookahead: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare the feature matrix and target labels.

    The target label is 1 if the future return over `lookahead` steps
    is positive, and 0 otherwise.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing features and the 'Close' column.
    feature_cols : tuple[str, ...]
        Names of the feature columns.
    lookahead : int
        Number of steps ahead to compute the future return.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix X and binary labels y.
    """
    df = df.copy()
    df["future_return"] = df["Close"].shift(-lookahead) / df["Close"] - 1.0
    df["label"] = (df["future_return"] > 0).astype(int)
    df.dropna(inplace=True)
    X = df[list(feature_cols)].values
    y = df["label"].values
    return X, y


def train_model(
    X: np.ndarray, y: np.ndarray, test_fraction: float = 0.2
) -> Tuple[CatBoostClassifier, float]:
    """Train a CatBoost classifier and compute AUC on the test set.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary labels.
    test_fraction : float, optional
        Fraction of the data reserved for testing (default is 0.2).

    Returns
    -------
    Tuple[CatBoostClassifier, float]
        Trained model and the AUC score on the test set.
    """
    from sklearn.metrics import roc_auc_score  # sklearn is not installed; provide fallback
    # Since sklearn is not available, implement a simple AUC manually
    def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute the AUC using the Mann–Whitney U statistic.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels (0 or 1).
        y_scores : np.ndarray
            Predicted probabilities for the positive class.

        Returns
        -------
        float
            Estimated AUC.
        """
        # Rank the scores
        order = np.argsort(y_scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(y_scores)) + 1
        # Sum of ranks for the positive class
        sum_ranks_pos = ranks[y_true == 1].sum()
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5  # not meaningful
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return auc

    # Split the data chronologically (no shuffle)
    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    # Train CatBoostClassifier.  Use simple parameters to reduce computation time.
    model = CatBoostClassifier(
        loss_function="Logloss",
        depth=4,
        iterations=200,
        learning_rate=0.1,
        verbose=False,
    )
    model.fit(X_train, y_train)
    # Predict probabilities on the test set
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = compute_auc(y_test, y_proba)
    return model, auc


def backtest_strategy(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    feature_cols: Tuple[str, ...],
    params: BacktestParameters,
) -> float:
    """Execute a simple trading strategy based on model probabilities.

    A position of +1 (long) is taken when the predicted probability is
    above `buy_threshold`.  A position of –1 (short) is taken when the
    probability is below `sell_threshold`.  Positions are held for
    `hold_period` steps before being closed.  The strategy is run
    sequentially over the data without lookahead bias beyond the
    specified `lookahead` used for feature construction.

    Parameters
    ----------
    model : CatBoostClassifier
        Trained classification model with a `predict_proba` method.
    df : pandas.DataFrame
        DataFrame containing features and the 'Close' column.
    feature_cols : tuple[str, ...]
        Names of the feature columns.
    params : BacktestParameters
        Configuration specifying thresholds and hold period.

    Returns
    -------
    float
        Final portfolio value assuming a starting value of 1.0.
    """
    features = df[list(feature_cols)].values
    proba = model.predict_proba(features)[:, 1]
    # Compute log returns for the asset
    asset_returns = df["Close"].pct_change().fillna(0).values
    position = 0  # +1 for long, –1 for short, 0 for flat
    time_in_position = 0
    portfolio_value = 1.0
    for i in range(len(df)):
        # Update portfolio based on previous position
        if i > 0:
            portfolio_value *= 1 + position * asset_returns[i]
            time_in_position += 1 if position != 0 else 0
        # Close position if hold period reached
        if position != 0 and time_in_position >= params.hold_period:
            position = 0
            time_in_position = 0
        # Generate new signal (avoid lookahead: do not use the last `lookahead` rows)
        if i < len(df) - params.lookahead:
            prob = proba[i]
            if position == 0:
                if prob > params.buy_threshold:
                    position = 1
                    time_in_position = 0
                elif prob < params.sell_threshold:
                    position = -1
                    time_in_position = 0
    return portfolio_value


def run_backtest(params: BacktestParameters) -> None:
    """Orchestrate the entire workflow and print results."""
    # Generate synthetic price data
    prices = generate_price_series(params)
    # Compute technical indicators
    data = compute_features(prices)
    feature_cols = ("sma_10", "sma_30", "rsi_14", "z_score_50")
    X, y = prepare_dataset(data, feature_cols, params.lookahead)
    model, auc = train_model(X, y)
    final_value = backtest_strategy(model, data.iloc[:-params.lookahead], feature_cols, params)
    print(f"Synthetic backtest completed over {params.n_steps} steps.")
    print(f"AUC on held‑out data: {auc:.4f}")
    print(f"Final portfolio value: {final_value:.4f}")


if __name__ == "__main__":
    # Default parameters; adjust if running interactively
    parameters = BacktestParameters()
    run_backtest(parameters)