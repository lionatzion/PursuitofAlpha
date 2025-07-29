"""
backtest_pipeline.py
Run backtest with trained model.
"""
import joblib
from modules.backtesting_workflow import run_backtest

def main():
    model=joblib.load("../models/trained/model.joblib")
    # df=...
    result=run_backtest(model,None)
    print("Final:",result)

if __name__=="__main__":
    main()
