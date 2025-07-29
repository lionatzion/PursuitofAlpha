"""
data_ingestion.py
Functions to download data from yfinance.
"""
import yfinance as yf

def download_data(tickers, start, end, interval="1h"):
    data={}
    for t in tickers:
        df=yf.download(t, start=start, end=end, interval=interval, progress=False)
        if not df.empty:
            df.dropna(inplace=True)
            data[t]=df
    return data
