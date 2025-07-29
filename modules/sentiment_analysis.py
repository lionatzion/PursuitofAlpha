"""
sentiment_analysis.py
Load FinBERT & score text.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def load_finbert(model_dir):
    model=AutoModelForSequenceClassification.from_pretrained(model_dir)
    tok=AutoTokenizer.from_pretrained(model_dir)
    return pipeline("sentiment-analysis", model=model, tokenizer=tok)

def score_sentiment(texts, pipe):
    return [pipe(t)[0]['label'] for t in texts]
