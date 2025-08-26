"""
data_ingestion.py
Robust helpers to download market data (Yahoo Finance) with basic validation.
"""
from __future__ import annotations
from typing import Dict, Iterable, Literal
import datetime as dt

import pandas as pd
import yfinance as yf


def _to_datestr(d: str | dt.date | dt.datetime) -> str:
    if isinstance(d, str):
        return d
    if isinstance(d, dt.datetime):
        return d.strftime("%Y-%m-%d")
    return d.strftime("%Y-%m-%d")


def download_data(
    tickers: Iterable[str],
    start: str | dt.date | dt.datetime,
    end: str | dt.date | dt.datetime,
    interval: Literal["1m","2m","5m","15m","30m","1h","90m","1d","1wk","1mo"] = "1h",
    auto_adjust: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data for given tickers using yfinance.

    Notes
    - Intraday data (<=1h) is limited to ~730 days by Yahoo; validate your date range.
    - Returns dict[ticker] -> DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
    """
    s = _to_datestr(start)
    e = _to_datestr(end)
    out: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        df = yf.download(
            t, start=s, end=e, interval=interval, progress=False, auto_adjust=auto_adjust
        )
        # Normalize columns if MultiIndex arrives
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] for c in df.columns]
        df = df.dropna(how="any")
        if not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.index.name = "Datetime"
            out[t] = df
    return out
