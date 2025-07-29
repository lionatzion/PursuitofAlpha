"""
model_training.py
Train ML models & save.
"""
import joblib
from sklearn.ensemble import GradientBoostingClassifier

def prepare_features(df, cols, lookahead=3):
    return None, None

def train_classifier(X, y):
    return None, 0.0

def save_model(model, path):
    joblib.dump(model, path)
