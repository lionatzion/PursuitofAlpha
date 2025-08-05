

import joblib
from modules.data_ingestion import download_data
from modules.feature_engineering import add_indicators
from modules.model_training import prepare_features
from modules.model_evaluation import evaluate_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def main():
    model = joblib.load("models/gradient_boost_model.joblib")
    tickers = ["AAPL", "MSFT"]  # Example: evaluate multiple tickers
    start_date = "2025-01-01"
    end_date = "2025-08-01"
    metrics_list = []
    for ticker in tickers:
        data = download_data([ticker], start_date, end_date)
        if ticker not in data or data[ticker].empty:
            print(f"No data for {ticker} in range {start_date} to {end_date}.")
            continue
        df = data[ticker]
        df = add_indicators(df)
        feature_cols = ["sma_10", "sma_30", "rsi_14", "z_score_50"]
        X, y = prepare_features(df, feature_cols)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        cm = confusion_matrix(y, y_pred)

        print(f"\nResults for {ticker}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{ticker} ROC (area = {auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {ticker}")
        plt.legend(loc="lower right")
        plt.show()

        # Save metrics for this ticker
        metrics_list.append({
            "ticker": ticker,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc
        })

        # Optionally call your custom evaluation
        evaluate_model(model, X, y)

    # Save all metrics to CSV
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        os.makedirs("metrics", exist_ok=True)
        metrics_df.to_csv("metrics/evaluation_metrics.csv", index=False)
        print("\nSaved evaluation metrics to metrics/evaluation_metrics.csv")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()