import streamlit as st
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = 'models/xgb_volatility.joblib'

@st.cache_data
def load_features(path='data/processed/features.parquet'):
    return pd.read_parquet(path)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("📈 Cryptocurrency Volatility Forecast")
st.write("Forecast next-day volatility using engineered features and XGBoost. Choose between using the latest data or entering your own scenario.")

# Load data and model
feat = load_features()
model = load_model()

# Mode selection
mode = st.radio("Choose input mode", ["Auto (latest data)", "Manual input"])


if mode == "Auto (latest data)":
    crypto_name = st.selectbox("Select cryptocurrency", sorted(feat['crypto_name'].unique()))
    crypto_name_df = feat[feat['crypto_name'] == crypto_name].sort_values('date').copy()

    if crypto_name_df.empty: 
        st.error(f"No data available for {crypto_name}. Please select another cryptocurrency.") 
        st.stop()
    latest = crypto_name_df.iloc[-1]
    

    X_latest = latest[['ret_1d','vol_7d','vol_14d','ma_7','ma_14','ma_30','ma_slope_14',
                       'parkinson','garman_klass','atr','liq','vol_ma_7','vol_ma_14','vol_vol_14',
                       'gap','hl_spread','dow','month','is_weekend']].to_frame().T
    X_latest['crypto_name'] = crypto_name

else:
    st.subheader("🔧 Enter your own feature values")

    crypto_name = st.selectbox("Cryptocurrency", sorted(feat['crypto_name'].unique()))
    ret_1d = st.number_input("Return (1D)", value=0.0)
    vol_7d = st.number_input("Volatility (7D)", value=0.01)
    vol_14d = st.number_input("Volatility (14D)", value=0.02)
    ma_7 = st.number_input("MA 7", value=100.0)
    ma_14 = st.number_input("MA 14", value=100.0)
    ma_30 = st.number_input("MA 30", value=100.0)
    ma_slope_14 = st.number_input("MA Slope 14", value=0.0)
    parkinson = st.number_input("Parkinson Volatility", value=0.01)
    garman_klass = st.number_input("Garman–Klass Volatility", value=0.01)
    atr = st.number_input("ATR", value=0.01)
    liq = st.number_input("Liquidity", value=0.01)
    vol_ma_7 = st.number_input("Volume MA 7", value=1_000_000.0)
    vol_ma_14 = st.number_input("Volume MA 14", value=1_000_000.0)
    vol_vol_14 = st.number_input("Volume Volatility 14", value=0.01)
    gap = st.number_input("Gap", value=0.01)
    hl_spread = st.number_input("High-Low Spread", value=0.01)
    dow = st.selectbox("Day of Week (0=Mon)", list(range(7)))
    month = st.selectbox("Month", list(range(1, 13)))
    is_weekend = st.selectbox("Is Weekend", [0, 1])

    X_latest = pd.DataFrame([{
        'crypto_name': crypto_name,
        'ret_1d': ret_1d,
        'vol_7d': vol_7d,
        'vol_14d': vol_14d,
        'ma_7': ma_7,
        'ma_14': ma_14,
        'ma_30': ma_30,
        'ma_slope_14': ma_slope_14,
        'parkinson': parkinson,
        'garman_klass': garman_klass,
        'atr': atr,
        'liq': liq,
        'vol_ma_7': vol_ma_7,
        'vol_ma_14': vol_ma_14,
        'vol_vol_14': vol_vol_14,
        'gap': gap,
        'hl_spread': hl_spread,
        'dow': dow,
        'month': month,
        'is_weekend': is_weekend
    }])

# Prediction
pred = model.predict(X_latest)[0]
st.metric("📊 Predicted next-day volatility (GK)", f"{pred:.4f}")

# Risk regime (only for auto mode)
# Risk regime (only for auto mode)
if mode == "Auto (latest data)":
    hist = crypto_name_df['garman_klass'].dropna()

    if hist.empty:
        st.warning(f"No historical GK volatility data available for {crypto_name}. Risk regime cannot be computed.")
    else:
        threshold = np.percentile(hist, 80)
        risk = "High" if pred > threshold else "Normal"
        st.write(f"**Risk regime:** {risk} (80th percentile threshold: {threshold:.4f})")

        st.line_chart(crypto_name_df[['date','garman_klass']].set_index('date'))


st.caption("Volatility proxy: Garman–Klass. Prediction is for next day.")
