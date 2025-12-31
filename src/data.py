import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_csv(path)

    # Drop unnecessary index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    # Rename lowercase versions
    df = df.rename(columns={
        'marketcap': 'market_cap'
    })

    # Ensure required columns exist
    required = ['crypto_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'market_cap']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert date
    df['date'] = pd.to_datetime(df['date'])

    # Sort and deduplicate
    df = df.sort_values(['crypto_name', 'date']).drop_duplicates(['crypto_name', 'date'])

    # Convert numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume', 'market_cap']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clip negative values
    df[['volume', 'market_cap']] = df[['volume', 'market_cap']].clip(lower=0)

    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    def _ffill(g):
        g = g.copy()
        g[['open','high','low','close','volume','market_cap']] = \
            g[['open','high','low','close','volume','market_cap']].ffill().bfill()
        return g
    return df.groupby('crypto_name', group_keys=False).apply(_ffill)

def winsorize_series(s, lower=0.01, upper=0.99):
    ql, qu = s.quantile([lower, upper])
    return s.clip(ql, qu)

def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['open','high','low','close','volume','market_cap']:
        df[col] = df.groupby('crypto_name')[col].transform(winsorize_series)
    return df
