# train.py
import pandas as pd
from src.data import load_data, fill_missing, clean_outliers
from src.features import build_features
from src.models import train_and_eval, save_model

raw = load_data('C:\\Users\\Suraj\\OneDrive\\Desktop\\Cryptocurrency Volatility Prediction ML Project\\data\\dataset.csv')
raw = fill_missing(raw)
raw = clean_outliers(raw)

feat = build_features(raw)
feat.to_parquet('data/processed/features.parquet')

pipe, metrics = train_and_eval(feat)
print(metrics)
save_model(pipe)
metrics.to_csv('reports/metrics.csv', index=False)
