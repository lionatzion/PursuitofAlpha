"""
train_pipeline.py
Load → feature → train → save.
"""
from modules.data_ingestion import download_data
from modules.feature_engineering import add_indicators, create_volume_bars
from modules.model_training import train_classifier, save_model

def main():
    tickers=["AAPL"]
    data=download_data(tickers,"2019-01-01","2024-12-31")
    # ...
    save_model(None,"../models/trained/model.joblib")

if __name__=="__main__":
    main()
