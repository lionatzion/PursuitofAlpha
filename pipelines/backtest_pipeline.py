"""
backtest_pipeline.py
Run a backtest using the trained GradientBoostingClassifier and recent Yahoo equity data.
"""
import joblib
from modules.data_ingestion import download_data
from modules.feature_engineering import add_indicators
from modules.model_training import prepare_features
from modules.backtesting_workflow import run_backtest
from datetime import datetime, timedelta

def main():
    model = joblib.load("models/gradient_boost_model.joblib")
    tickers = ["AAPL"]
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)  # last 6 months
    data = download_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    df = data[tickers[0]]
    df = add_indicators(df)
    feature_cols = ["sma_10", "sma_30", "rsi_14", "z_score_50"]
    X, y = prepare_features(df, feature_cols)
    final_value = run_backtest(model, df)
    print(f"Final portfolio value: {final_value:.2f}")

if __name__ == "__main__":
    main()