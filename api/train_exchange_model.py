import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score
from supabase import create_client, Client

# Add parent directory to path for potential imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import some helpers from stock_ai (if possible) or redefine them to be standalone
# Redefining to ensure standalone functionality in CI/CD
def _finite_float(value):
    try:
        v = float(value)
    except Exception:
        return None
    return v if np.isfinite(v) else None

def add_technical_indicators(df):
    cols = {c.lower(): c for c in df.columns}
    close_col = cols.get("close")
    open_col = cols.get("open")
    high_col = cols.get("high")
    low_col = cols.get("low")
    volume_col = cols.get("volume")
    
    if not close_col or not volume_col:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["Close"] = df[close_col]
    out["Volume"] = df[volume_col]
    if open_col: out["Open"] = df[open_col]
    if high_col: out["High"] = df[high_col]
    if low_col: out["Low"] = df[low_col]
    
    # 1. Moving Averages
    out["SMA_50"] = out["Close"].rolling(window=50, min_periods=1).mean()
    out["SMA_200"] = out["Close"].rolling(window=200, min_periods=1).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["EMA_200"] = out["Close"].ewm(span=200, adjust=False).mean()
    
    # 2. MACD
    ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_12 - ema_26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    
    # 3. RSI
    delta = out["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    
    # 4. Momentum & ROC
    out["Momentum"] = out["Close"].pct_change().fillna(0)
    out["ROC_12"] = out["Close"].pct_change(periods=12).fillna(0) * 100
    
    # 5. Volume Indicators
    out["VOL_SMA20"] = out["Volume"].rolling(window=20, min_periods=1).mean()
    
    # 6. Advanced (Requires High/Low)
    if "High" in out.columns and "Low" in out.columns:
        high = out["High"].astype(float)
        low = out["Low"].astype(float)
        close = out["Close"].astype(float)
        prev_close = close.shift(1)
        
        # ATR
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        out["ATR_14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=1).mean()
        
        # ADX
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr_sm = tr.ewm(alpha=1/14, adjust=False, min_periods=1).mean()
        plus_dm_sm = pd.Series(plus_dm, index=out.index).ewm(alpha=1/14, adjust=False, min_periods=1).mean()
        minus_dm_sm = pd.Series(minus_dm, index=out.index).ewm(alpha=1/14, adjust=False, min_periods=1).mean()
        
        plus_di = 100 * (plus_dm_sm / tr_sm.replace(0.0, np.nan))
        minus_di = 100 * (minus_dm_sm / tr_sm.replace(0.0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
        out["ADX_14"] = dx.ewm(alpha=1/14, adjust=False, min_periods=1).mean().fillna(0.0)
        
        # Stochastic
        lowest_low = low.rolling(window=14, min_periods=1).min()
        highest_high = high.rolling(window=14, min_periods=1).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0.0, np.nan)
        out["STOCH_K"] = stoch_k.fillna(0.0)
        out["STOCH_D"] = out["STOCH_K"].rolling(window=3, min_periods=1).mean()
        
        # CCI
        tp = (high + low + close) / 3
        tp_sma = tp.rolling(window=20, min_periods=1).mean()
        mean_dev = (tp - tp_sma).abs().rolling(window=20, min_periods=1).mean()
        out["CCI_20"] = ((tp - tp_sma) / (0.015 * mean_dev.replace(0.0, np.nan))).fillna(0.0)
        
        # VWAP (Rolling 20-day approx)
        pv = tp * out["Volume"].astype(float)
        vol_sum = out["Volume"].astype(float).rolling(window=20, min_periods=1).sum()
        out["VWAP_20"] = (pv.rolling(window=20, min_periods=1).sum() / vol_sum.replace(0.0, np.nan)).fillna(0.0)
    else:
        # Fill zeros if OHLC not fully available
        for c in ["ATR_14", "ADX_14", "STOCH_K", "STOCH_D", "CCI_20", "VWAP_20"]:
            out[c] = 0.0
            
    return out

def prepare_for_ai(df):
    if df.empty: return df
    out = df.copy()
    out["Next_Close"] = out["Close"].shift(-1)
    out["Target"] = (out["Next_Close"] > out["Close"]).astype(int)
    out = out.dropna().copy()
    return out

def train_model(exchange, supabase_url, supabase_key):
    print(f"Starting training for exchange: {exchange}")
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # 1. Fetch symbols for this exchange
    res = supabase.table("stock_prices").select("symbol").eq("exchange", exchange).execute()
    if not res.data:
        print(f"No price data found for exchange {exchange}")
        return
    
    all_symbols = sorted(list(set(r["symbol"] for r in res.data)))
    print(f"Found {len(all_symbols)} symbols for exchange {exchange}")
    
    # 2. Collect data from a subset of symbols (to avoid memory issues and too long training)
    # Sampling up to 20 symbols if there are too many
    sample_symbols = all_symbols[:30] 
    
    combined_data = []
    
    for symbol in sample_symbols:
        print(f"Fetching data for {symbol}...")
        res = supabase.table("stock_prices").select("*").eq("symbol", symbol).eq("exchange", exchange).order("date", desc=False).execute()
        if not res.data or len(res.data) < 200:
            continue
        
        df = pd.DataFrame(res.data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        
        feat = add_technical_indicators(df)
        ready = prepare_for_ai(feat)
        
        if len(ready) >= 120:
            combined_data.append(ready)
            
    if not combined_data:
        print("No valid data collected for training")
        return

    df_train = pd.concat(combined_data)
    print(f"Total training samples: {len(df_train)}")
    
    # 3. Train
    predictors = [
        "Close", "Volume", 
        "SMA_50", "SMA_200", 
        "EMA_50", "EMA_200", 
        "RSI", "Momentum", "ROC_12", 
        "MACD", "MACD_Signal",
        "ATR_14", "ADX_14",
        "STOCH_K", "STOCH_D",
        "CCI_20", "VWAP_20", "VOL_SMA20"
    ]
    model = LGBMClassifier(n_estimators=100, random_state=1, n_jobs=-1, verbose=-1)
    model.fit(df_train[predictors], df_train["Target"])
    
    # 4. Save and Upload
    filename = f"model_{exchange}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filename}")
    
    # Upload to Supabase Storage (bucket 'ai-models')
    try:
        with open(filename, "rb") as f:
            supabase.storage.from_("ai-models").upload(
                path=filename,
                file=f,
                file_options={"cache-control": "3600", "upsert": "true"}
            )
        print(f"Model successfully uploaded to Supabase Storage: ai-models/{filename}")
    except Exception as e:
        print(f"Failed to upload model: {e}")
        # If bucket doesn't exist, this might fail. In reality, the user should create it.
        # But we can try to log it clearly.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", required=True, help="Exchange name to train on")
    args = parser.parse_args()
    
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("Error: NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required.")
        sys.exit(1)
        
    train_model(args.exchange, url, key)
