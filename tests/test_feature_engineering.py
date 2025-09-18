import pandas as pd
import numpy as np

from modules.feature_engineering import create_volume_bars, add_indicators


def test_create_volume_bars_basic():
    # 2 bars expected with vol_target=700k: (300k+400k), (500k+200k)
    df = pd.DataFrame(
        {
            "Open": [10, 11, 12, 13],
            "High": [11, 12, 13, 14],
            "Low": [9, 10, 11, 12],
            "Close": [10.5, 11.5, 12.5, 13.5],
            "Volume": [300_000, 400_000, 500_000, 200_000],
        }
    )
    bars = create_volume_bars(df, vol_target=700_000)
    assert len(bars) == 2
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(bars.columns)
    assert bars.index.name == "Bar"
    # Each bar volume should meet or exceed target
    assert (bars["Volume"] >= 700_000).all()


def test_add_indicators_outputs_columns_and_non_null():
    n = 120
    close = np.linspace(100, 120, n)
    df = pd.DataFrame({"Close": close})
    out = add_indicators(df, sma_fast=10, sma_slow=30, rsi_window=14, z_window=50)
    expected = {"sma_10", "sma_30", "rsi_14", "z_score_50"}
    assert expected.issubset(out.columns)
    assert out[sorted(expected)].isna().sum().sum() == 0

