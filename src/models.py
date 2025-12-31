import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

FEATURES_NUM = [
    'ret_1d','vol_7d','vol_14d','ma_7','ma_14','ma_30','ma_slope_14',
    'parkinson','garman_klass','atr','liq','vol_ma_7','vol_ma_14','vol_vol_14',
    'gap','hl_spread'
]
FEATURES_CAT = ['crypto_name','dow','month','is_weekend']

def make_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), FEATURES_NUM),
            ('cat', OneHotEncoder(handle_unknown='ignore'), FEATURES_CAT)
        ]
    )
    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline(steps=[('pre', pre), ('model', model)])
    return pipe

def time_series_split(df: pd.DataFrame, test_frac=0.15, val_frac=0.15):
    df = df.sort_values(['date'])
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n - n_test - n_val]
    val = df.iloc[n - n_test - n_val: n - n_test]
    test = df.iloc[n - n_test:]
    return train, val, test

def train_and_eval(feat: pd.DataFrame):
    cols = FEATURES_NUM + FEATURES_CAT + ['target_vol','date']
    feat = feat[cols].dropna()
    train, val, test = time_series_split(feat)

    X_train = train[FEATURES_NUM + FEATURES_CAT]
    y_train = train['target_vol']
    X_val = val[FEATURES_NUM + FEATURES_CAT]
    y_val = val['target_vol']
    X_test = test[FEATURES_NUM + FEATURES_CAT]
    y_test = test['target_vol']

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    def eval_split(X, y, name):
        pred = pipe.predict(X)
        return {
            'split': name,
            'RMSE': float(np.sqrt(((pred - y) ** 2).mean())),
            'MAE': float(mean_absolute_error(y, pred)),
            'R2': float(r2_score(y, pred))
        }

    metrics = [
        eval_split(X_train, y_train, 'train'),
        eval_split(X_val, y_val, 'val'),
        eval_split(X_test, y_test, 'test')
    ]
    return pipe, pd.DataFrame(metrics)

def save_model(pipe, path='models/xgb_volatility.joblib'):
    joblib.dump(pipe, path)
