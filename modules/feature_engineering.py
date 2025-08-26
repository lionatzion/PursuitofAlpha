"""
feature_engineering.py
Volume bars plus common technical indicators.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ta import volatility  # for RSI convenience


def create_volume_bars(df: pd.DataFrame, vol_target: float = 1e6) -> pd.DataFrame:
    """
    Aggregate rows into bars that each contain ~vol_target total traded volume.
    Expects df with columns: Open, High, Low, Close, Volume
    """
    if df.empty:
        return df

    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[cols].copy()

    bars = []
    cum_vol = 0.0
    o = h = l = c = None
    for _, row in df.iterrows():
        if cum_vol == 0:
            o = row["Open"]
            h = row["High"]
            l = row["Low"]
        h = max(h, row["High"])
        l = min(l, row["Low"])
        c = row["Close"]
        cum_vol += float(row["Volume"])

        if cum_vol >= vol_target:
            bars.append({"Open": o, "High": h, "Low": l, "Close": c, "Volume": cum_vol})
            cum_vol = 0.0

    vol_df = pd.DataFrame(bars)
    if not vol_df.empty:
        vol_df.index.name = "Bar"
    return vol_df


def add_indicators(
    df: pd.DataFrame, sma_fast: int = 10, sma_slow: int = 30, rsi_window: int = 14, z_window: int = 50
) -> pd.DataFrame:
    """
    Add SMA, RSI, Z-score. Returns a copy with new columns:
    sma_{fast}, sma_{slow}, rsi_{window}, z_score_{z_window}
    """
    if df.empty:
        return df

    out = df.copy()
    out[f"sma_{sma_fast}"] = out["Close"].rolling(window=sma_fast).mean()
    out[f"sma_{sma_slow}"] = out["Close"].rolling(window=sma_slow).mean()
    out[f"rsi_{rsi_window}"] = volatility.rsi(out["Close"], window=rsi_window)

    roll_mean = out["Close"].rolling(window=z_window).mean()
    roll_std = out["Close"].rolling(window=z_window).std()
    out[f"z_score_{z_window}"] = (out["Close"] - roll_mean) / roll_std

    out = out.dropna(how="any")
    return out
