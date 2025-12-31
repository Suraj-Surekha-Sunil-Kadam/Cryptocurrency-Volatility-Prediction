import numpy as np
import pandas as pd

def add_returns(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g['log_ret'] = np.log(g['close']).diff()
    g['ret_1d'] = g['close'].pct_change()
    for w in [7,14,30]:
        g[f'vol_{w}d'] = g['log_ret'].rolling(w).std()
        g[f'ma_{w}'] = g['close'].rolling(w).mean()
    g['ma_slope_14'] = g['ma_{w}'] if False else g['close'].rolling(14).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    return g

def add_hl_vol(g: pd.DataFrame, W=14) -> pd.DataFrame:
    g = g.copy()
    hl = np.log(g['high']/g['low'])
    oc = np.log(g['close']/g['open']).fillna(0)
    g['parkinson'] = (hl**2).rolling(W).mean().apply(lambda x: np.sqrt(x/(4*np.log(2))))
    g['garman_klass'] = (0.5*hl**2 - 2*oc**2).rolling(W).mean().apply(np.sqrt)
    return g

def add_atr(g: pd.DataFrame, W=14) -> pd.DataFrame:
    g = g.copy()
    prev_close = g['close'].shift(1)
    tr = pd.concat([
        g['high'] - g['low'],
        (g['high'] - prev_close).abs(),
        (g['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    g['atr'] = tr.rolling(W).mean()
    return g

def add_liquidity(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g['liq'] = g['volume'] / g['market_cap'].replace(0, np.nan)
    for w in [7,14,30]:
        g[f'vol_ma_{w}'] = g['volume'].rolling(w).mean()
        g[f'vol_vol_{w}'] = g['volume'].rolling(w).std()
    return g

def add_calendar(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g['dow'] = g['date'].dt.dayofweek
    g['month'] = g['date'].dt.month
    g['is_weekend'] = (g['dow'] >= 5).astype(int)
    g['gap'] = (g['open']/g['close'].shift(1) - 1).fillna(0)
    g['hl_spread'] = np.log(g['high']/g['low'])
    return g

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    def _per_symbol(g):
        g = add_returns(g)
        g = add_hl_vol(g)
        g = add_atr(g)
        g = add_liquidity(g)
        g = add_calendar(g)
        # Target: next-day GK volatility
        g['target_vol'] = g['garman_klass'].shift(-1)
        return g
    feat = df.groupby('crypto_name', group_keys=False).apply(_per_symbol)
    feat = feat.dropna(subset=['target_vol'])
    return feat
