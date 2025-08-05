"""
train_pipeline.py
Train a GradientBoostingClassifier on Yahoo equity data for the last 3 years.
"""
from modules.data_ingestion import download_data
from modules.feature_engineering import add_indicators
from modules.model_training import prepare_features, train_classifier, save_model
import os
from datetime import datetime, timedelta

def main():
    tickers = ["AAPL"]
    end_date = datetime.today()
    start_date = end_date - timedelta(days=3*365)
    data = download_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    df = data[tickers[0]]
    df = add_indicators(df)
    feature_cols = ["sma_10", "sma_30", "rsi_14", "z_score_50"]
    X, y = prepare_features(df, feature_cols)
    model, auc = train_classifier(X, y)
    print(f"Trained model ROC AUC: {auc:.4f}")
    os.makedirs("models", exist_ok=True)
    save_model(model, "models/gradient_boost_model.joblib")

if __name__ == "__main__":
    main()