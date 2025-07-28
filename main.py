from modules import backtesting_workflow as bw
import os
import joblib

if __name__ == "__main__":
    tickers = ["AAPL"]
    data = bw.download_data(tickers, start="2019-01-01", end="2024-12-31", interval="1h")
    if not data:
        raise SystemExit("No data downloaded.")

    df = bw.create_volume_bars(data[tickers[0]], vol_target=5e6)
    df = bw.add_indicators(df, sma_fast=10, sma_slow=30, rsi_window=14, z_window=50)
    feature_cols = ["sma_10", "sma_30", "rsi_14", "z_score_50"]
    X, y = bw.prepare_features(df, feature_cols)

    model, auc = bw.train_classifier(X, y)
    print(f"Trained model ROC AUC: {auc:.4f}")

    # Save model
    model_path = os.path.join("models", "gradient_boost_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    final = bw.run_backtest(model, df)
    print(f"Final portfolio value: {final:.2f}")