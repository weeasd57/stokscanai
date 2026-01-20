import os
import sys
import pickle

# Add api to path
api_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api")
sys.path.insert(0, api_dir)

from dotenv import load_dotenv
load_dotenv()

from stock_ai import (
    _get_exchange_bulk_data,
    _get_data_with_indicators_cached,
    add_technical_indicators,
    prepare_for_ai,
    _ensure_feature_columns,
)
from routers.scan_ai_fast import _load_model

# Test params
model_name = "SUPER_EGX_EXTENDED_RANDOM_3000.pkl"
exchange = "EGX"
symbol = "COMI"  # Example EGX symbol

# Load model
model_entry = _load_model(model_name)
if not model_entry:
    print(f"Model {model_name} not loaded")
    sys.exit(1)
model, predictors, is_lgbm = model_entry
print(f"Model loaded. Predictors: {len(predictors)}, is_lgbm: {is_lgbm}")

# Load data
bulk_data = _get_exchange_bulk_data(exchange, from_date="2020-01-01")
print(f"Bulk data loaded for {len(bulk_data)} symbols")

if symbol not in bulk_data:
    print(f"Symbol {symbol} not in bulk data")
    sys.exit(1)

df = bulk_data[symbol]
print(f"Raw data for {symbol}: {len(df)} rows")

if len(df) > 500:
    df = df.iloc[-500:].copy()

# Add indicators
feat = _get_data_with_indicators_cached(symbol, exchange, df, add_technical_indicators)
print(f"After indicators: {len(feat)} rows, columns: {feat.columns.tolist()[:10]}...")

# Prepare for AI
candidate = prepare_for_ai(feat)
print(f"After prepare_for_ai: {len(candidate)} rows")
print(f"Last date in candidate: {candidate.index[-1] if len(candidate) > 0 else 'N/A'}")
print(f"Last date in raw data: {df.index[-1]}")

if len(candidate) < 60:
    print("Not enough rows after prepare_for_ai")
    sys.exit(1)

# Get last row
last_row = candidate.iloc[[-1]].copy()
_ensure_feature_columns(last_row, predictors)
available_predictors = [p for p in predictors if p in last_row.columns]
print(f"Available predictors: {len(available_predictors)}")

# Predict
pred = int(model.predict(last_row[available_predictors])[0])
print(f"Prediction: {pred} ({'BUY' if pred == 1 else 'SELL'})")

# Get probability
if hasattr(model, "predict_proba"):
    try:
        prob = float(model.predict_proba(last_row[available_predictors])[0][1])
        print(f"Probability (BUY): {prob:.4f}")
    except Exception as e:
        print(f"predict_proba error: {e}")
