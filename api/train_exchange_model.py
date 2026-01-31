import os
import warnings
import time

# Suppress specific FutureWarnings from libraries like 'ta'
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import argparse
import pickle
import json
from typing import Optional, Callable, Any, Dict, List

import pandas as pd
import numpy as np
from datetime import datetime
from ta import add_all_ta_features
import ta
from lightgbm import LGBMClassifier
import lightgbm as lgb
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from supabase import create_client, Client
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Memory

import tempfile

# Initialize memory cache for heavy feature engineering.
# IMPORTANT: keep it OUTSIDE the repo tree, otherwise uvicorn --reload will detect changes and restart mid-training.
_JOBLIB_CACHE_DIR = os.getenv("STOKSCANAI_JOBLIB_CACHE_DIR")
_DISABLE_JOBLIB_CACHE = os.getenv("STOKSCANAI_DISABLE_JOBLIB_CACHE", "0").strip().lower() in {"1", "true", "yes"}

if _DISABLE_JOBLIB_CACHE:
    memory_cache = Memory(location=None, verbose=0)
else:
    cache_dir = _JOBLIB_CACHE_DIR or os.path.join(tempfile.gettempdir(), "stokscanai_joblib_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        # If temp dir creation fails, fall back to no caching (prefer stability over speed)
        cache_dir = None
    memory_cache = Memory(location=cache_dir, verbose=0)

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import optuna
except ImportError:
    optuna = None

# Add parent directory to path for potential imports
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def _downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to save memory."""
    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
    return df

def _finite_float(value):
    try:
        v = float(value)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _write_training_summary(summary: dict) -> None:
    """Write last training summary to a local JSON file for UI consumption."""
    try:
        api_dir = os.path.dirname(os.path.abspath(__file__))
        summary_path = os.path.join(api_dir, "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f)
    except Exception as e:
        # Fail softly – training itself should not break because of logging issues
        print(f"Failed to write training summary: {e}")

class StreamCallback:
    def __init__(self, progress_cb):
        self.progress_cb = progress_cb
        self.history = []

    def __call__(self, env):
        # env.iteration, env.evaluation_result_list
        # evaluation_result_list example: [('valid_0', 'logloss', 0.54321, False)]
        try:
            iteration = env.iteration
            metrics = {}
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                key = f"{data_name}_{eval_name}"
                metrics[key] = result
            
            # Construct a small payload for the graph
            point = {
                "iteration": iteration,
                **metrics
            }
            
            # Send to frontend
            if self.progress_cb:
                self.progress_cb({
                    "phase": "training_stream", 
                    "message": f"Training iter {iteration}", 
                    "stats": point
                })
        except Exception:
            pass


def fetch_fundamentals_for_exchange(supabase: Client, exchange: str) -> pd.DataFrame:
    """Fetch all fundamental data for a given exchange from stock_fundamentals table."""
    try:
        res = supabase.table("stock_fundamentals").select("symbol, data").eq("exchange", exchange).execute()
        if not res.data:
            return pd.DataFrame()
        
        funds = []
        for row in res.data:
            data = row.get("data", {})
            if not data: continue
            
            # Safety: Parse JSON if returned as string
            if isinstance(data, str):
                try:
                    import json
                    data = json.loads(data)
                except:
                    pass
            
            flat = {
                "symbol": row["symbol"],
                "marketCap": _finite_float(data.get("marketCap")),
                "peRatio": _finite_float(data.get("peRatio")),
                "eps": _finite_float(data.get("eps")),
                "dividendYield": _finite_float(data.get("dividendYield")),
                "sector": data.get("sector"),
                "industry": data.get("industry")
            }
            funds.append(flat)
            
        df_funds = pd.DataFrame(funds)
        # Force common fundamental columns to numeric if they exist to prevent LightGBM dtype errors
        for c in ["marketCap", "peRatio", "eps", "dividendYield"]:
            if c in df_funds.columns:
                df_funds[c] = pd.to_numeric(df_funds[c], errors='coerce')
        return df_funds
    except Exception as e:
        print(f"Warning: Failed to fetch fundamentals: {e}")
        return pd.DataFrame()



# Removed legacy optimize_and_train_model (GridSearchCV) in favor of Optuna-based optimization in ModelTrainer.

def add_indicator_signals(df):
    """
    Generate discrete signals (-1, 0, 1) from classical indicators.
    """
    df = df.copy()
    
    # Determine close column name (handle both cases)
    close_col = 'Close' if 'Close' in df.columns else 'close'
    
    # 1. RSI Signal (<30 Buy, >70 Sell)
    if 'RSI' not in df.columns:
        df['RSI'] = ta.momentum.rsi(df[close_col], window=14)
        
    df['feat_rsi_signal'] = 0
    df.loc[df['RSI'] < 30, 'feat_rsi_signal'] = 1
    df.loc[df['RSI'] > 70, 'feat_rsi_signal'] = -1
    
    # 2. EMA Cross (Golden Cross)
    if 'EMA_50' not in df.columns:
        df['EMA_50'] = df[close_col].ewm(span=50).mean()
    if 'EMA_200' not in df.columns:
        df['EMA_200'] = df[close_col].ewm(span=200).mean()
        
    df['feat_ema_signal'] = 0
    df.loc[df['EMA_50'] > df['EMA_200'], 'feat_ema_signal'] = 1
    df.loc[df['EMA_50'] < df['EMA_200'], 'feat_ema_signal'] = -1

    # 3. Bollinger Bands Signal (Close < Lower = Buy)
    if 'BB_Lower' not in df.columns or 'BB_Upper' not in df.columns:
         indicator_bb = ta.volatility.BollingerBands(close=df[close_col], window=20, window_dev=2)
         df['BB_Lower'] = indicator_bb.bollinger_lband()
         df['BB_Upper'] = indicator_bb.bollinger_hband()

    df['feat_bb_signal'] = 0
    df.loc[df[close_col] < df['BB_Lower'], 'feat_bb_signal'] = 1
    df.loc[df[close_col] > df['BB_Upper'], 'feat_bb_signal'] = -1
    
    return df

def add_rolling_win_rate(df, window=30):
    """
    Calculate rolling win rate for each indicator signal over the past 'window' days.
    Prevent Look-ahead bias by shifting results.
    """
    df = df.copy()
    
    # Determine close column name (handle both cases)
    close_col = 'Close' if 'Close' in df.columns else 'close' if 'close' in df.columns else None
    if not close_col:
        return df  # Cannot calculate without close column
    
    # Target: Did price go up tomorrow? (Shift -1)
    target_up = (df[close_col].shift(-1) > df[close_col]).astype(int)
    
    # Check correctness
    # Correct if (Signal=1 AND Up) OR (Signal=-1 AND Down)
    rsi_correct = (
        ((df['feat_rsi_signal'] == 1) & (target_up == 1)) | 
        ((df['feat_rsi_signal'] == -1) & (target_up == 0))
    ).astype(int)
    
    ema_correct = (
         ((df['feat_ema_signal'] == 1) & (target_up == 1)) | 
         ((df['feat_ema_signal'] == -1) & (target_up == 0))
    ).astype(int)
    
    bb_correct = (
         ((df['feat_bb_signal'] == 1) & (target_up == 1)) | 
         ((df['feat_bb_signal'] == -1) & (target_up == 0))
    ).astype(int)
    
    # Rolling Mean with Shift(1) to avoid data leakage
    # We use shift(1) because at time T, we can only measure if signal at T-1 was correct (which resolves at T).
    # Actually, signal at T-1 predicts T. So at T we look back. 
    # But to use it as a feature at T, we need "Past Accuracy".
    # Correctness of T-1 is available at T.
    # So we take rolling mean of correctness shifted by 1.
    df['feat_rsi_acc'] = rsi_correct.shift(1).rolling(window=window).mean().fillna(0.5)
    df['feat_ema_acc'] = ema_correct.shift(1).rolling(window=window).mean().fillna(0.5)
    df['feat_bb_acc'] = bb_correct.shift(1).rolling(window=window).mean().fillna(0.5)
    
    return df

def add_market_context(stock_df, market_df):
    """
    Add Market Context features (Trend, Volatility, Relative Strength).
    """
    if market_df is None or market_df.empty:
        # Return with 0s if no market data, to avoid crashing
        stock_df = stock_df.copy()
        stock_df['feat_mkt_trend'] = 0
        stock_df['feat_mkt_volatility'] = 0
        stock_df['feat_rel_strength'] = 0
        return stock_df

    # Ensure indexes are DatetimeIndex
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        stock_df.index = pd.to_datetime(stock_df.index)
    if not isinstance(market_df.index, pd.DatetimeIndex):
        market_df.index = pd.to_datetime(market_df.index)

    # Reindex market data to match stock data (forward fill)
    market_reindexed = market_df.reindex(stock_df.index, method='ffill')

    stock_df = stock_df.copy()
    
    # 1. Market Trend (is Market > SMA200?)
    # We calculate on the reindexed series to align with stock dates
    stock_df['mkt_close'] = market_reindexed['close']
    stock_df['mkt_sma200'] = stock_df['mkt_close'].rolling(200).mean()
    stock_df['feat_mkt_trend'] = 0
    stock_df.loc[stock_df['mkt_close'] > stock_df['mkt_sma200'], 'feat_mkt_trend'] = 1
    stock_df.loc[stock_df['mkt_close'] < stock_df['mkt_sma200'], 'feat_mkt_trend'] = -1
    
    # 2. Market Volatility (ATR) - Assuming market_reindexed has ATR or we compute rolling std
    # Simple proxy: Rolling std of returns
    stock_df['feat_mkt_volatility'] = stock_df['mkt_close'].pct_change().rolling(20).std().fillna(0)

    # 3. Relative Strength (Stock vs Market)
    stock_ret = stock_df['close'].pct_change()
    market_ret = stock_df['mkt_close'].pct_change()
    stock_df['feat_rel_strength'] = (stock_ret - market_ret).fillna(0)
    
    # Cleanup
    stock_df.drop(columns=['mkt_close', 'mkt_sma200'], inplace=True, errors='ignore')
    
    return stock_df

@memory_cache.cache
def add_massive_features(df):
    """
    Generate over 250 features using:
    1. TA library (~90 features)
    2. Rolling Stats
    3. Lagged historical data
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    for key in ("open", "high", "low", "close", "volume"):
        if key in cols and cols[key] != key:
            df.rename(columns={cols[key]: key}, inplace=True)
    if "close" not in df.columns or "volume" not in df.columns:
        return df
    
    # ---------------------------------------------------------
    # 1. Ready-made Technical Indicators (Base: ~80-90 features)
    # ---------------------------------------------------------
    # This step alone adds RSI, MACD, Bollinger, Ichimoku, etc.
    df = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )
    
    # 2. Vectorized Rolling Windows & Tags
    windows = [3, 7, 14, 30]
    target_cols = ['close', 'volume', 'momentum_rsi']
    extra_cols = {}
    
    # Pre-calculate rolling objects for each window to reuse
    for w in windows:
        existing = [c for c in target_cols if c in df.columns]
        if not existing: continue
            
        # Grouped rolling operation is faster than individual ones
        roll = df[existing].rolling(window=w)
        means = roll.mean()
        stds = roll.std()
        maxs = roll.max()
        mins = roll.min()
        
        for col in existing:
            extra_cols[f'{col}_SMA_{w}'] = means[col]
            extra_cols[f'{col}_STD_{w}'] = stds[col]
            extra_cols[f'{col}_MAX_{w}'] = maxs[col]
            extra_cols[f'{col}_MIN_{w}'] = mins[col]

    # 3. Historical Memory (Lag Features)
    # Detect correct column names (case-insensitive)
    close_col = 'close' if 'close' in df.columns else 'Close' if 'Close' in df.columns else None
    vol_col = 'volume' if 'volume' in df.columns else 'Volume' if 'Volume' in df.columns else None
    
    if close_col and vol_col:
        lags = [1, 2, 3, 5]
        for lag in lags:
            extra_cols[f'Close_Lag_{lag}'] = df[close_col].shift(lag)
            extra_cols[f'Vol_Lag_{lag}'] = df[vol_col].shift(lag)
            extra_cols[f'Return_{lag}d'] = df[close_col].pct_change(lag)

    # 4. Advanced Vectorized Math
    if close_col:
        extra_cols['Log_Ret'] = np.log(df[close_col] / df[close_col].shift(1).replace(0, np.nan))
        
        # Optimizing Z-Score (one rolling object)
        roll20 = df[close_col].rolling(20)
        mu20 = roll20.mean()
        sigma20 = roll20.std()
        extra_cols['Z_Score_20'] = (df[close_col] - mu20) / sigma20.replace(0, np.nan)
        
        if vol_col:
            extra_cols['PV_Trend'] = df[close_col].pct_change() * df[vol_col].pct_change()

    # Maintain Case-Sensitive Columns for other functions
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c.lower() in df.columns:
            extra_cols[c] = df[c.lower()]

    if extra_cols:
        df = pd.concat([df, pd.DataFrame(extra_cols, index=df.index)], axis=1)
        df = df.copy()

    # Clean data (remove NaN values resulting from Lags) - only for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

@memory_cache.cache
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

    # Cross features (Golden Cross / Death Cross logic)
    out["SMA_Cross"] = (out["SMA_50"] - out["SMA_200"]) / out["SMA_200"].replace(0, np.nan)
    out["EMA_Cross"] = (out["EMA_50"] - out["EMA_200"]) / out["EMA_200"].replace(0, np.nan)
    out["Price_vs_SMA200"] = (out["Close"] - out["SMA_200"]) / out["SMA_200"].replace(0, np.nan)
    
    # 2. MACD
    ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_12 - ema_26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
    
    # 3. RSI
    delta = out["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    
    # RSI 7 (Fast Momentum)
    gain7 = (delta.where(delta > 0, 0.0)).rolling(window=7).mean()
    loss7 = (-delta.where(delta < 0, 0.0)).rolling(window=7).mean()
    rs7 = gain7 / loss7.replace(0.0, np.nan)
    out["RSI_7"] = 100 - (100 / (1 + rs7))
    
    # 4. Momentum & ROC
    out["Momentum"] = out["Close"].pct_change().fillna(0)
    out["ROC_12"] = out["Close"].pct_change(periods=12).fillna(0) * 100
    
    # 5. Volume Indicators
    out["VOL_SMA20"] = out["Volume"].rolling(window=20, min_periods=1).mean()
    out["VOL_Change"] = out["Volume"].pct_change().fillna(0)
    
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

    # 7. Bollinger bands (20, 2) and derived features
    if "Close" in out.columns:
        bb_sma20 = out["Close"].rolling(window=20).mean()
        bb_std20 = out["Close"].rolling(window=20).std()
        out["BB_Upper"] = bb_sma20 + (2 * bb_std20)
        out["BB_Lower"] = bb_sma20 - (2 * bb_std20)

        width = out["BB_Upper"] - out["BB_Lower"]
        out["BB_PctB"] = (
            (out["Close"] - out["BB_Lower"]) / width.replace(0.0, np.nan)
        ).fillna(0.0)
        out["BB_Width"] = (
            width / out["Close"].replace(0.0, np.nan)
        ).fillna(0.0)
    else:
        out["BB_Upper"] = 0.0
        out["BB_Lower"] = 0.0
        out["BB_PctB"] = 0.0
        out["BB_Width"] = 0.0

    # 8. On-Balance Volume and slope
    price_delta = out["Close"].diff()
    direction = np.sign(price_delta).fillna(0.0)
    obv = (direction * out["Volume"]).cumsum()
    out["OBV"] = obv.fillna(0.0)
    out["OBV_Slope"] = out["OBV"].diff().fillna(0.0)

    # 9. Distance from rolling high/low for context
    rolling_high = out["Close"].rolling(window=100, min_periods=1).max()
    rolling_low = out["Close"].rolling(window=100, min_periods=1).min()
    out["Dist_From_High"] = (
        (out["Close"] / rolling_high.replace(0.0, np.nan)) - 1.0
    ).fillna(0.0)
    out["Dist_From_Low"] = (
        (out["Close"] / rolling_low.replace(0.0, np.nan)) - 1.0
    ).fillna(0.0)

    # 10. Z-score of price vs rolling mean/std
    rolling_mean = out["Close"].rolling(window=50, min_periods=1).mean()
    rolling_std = out["Close"].rolling(window=50, min_periods=1).std()
    out["Z_Score"] = (
        (out["Close"] - rolling_mean) / rolling_std.replace(0.0, np.nan)
    ).fillna(0.0)

    # 11. Candle geometry (body and shadows)
    if "Open" in out.columns and "High" in out.columns and "Low" in out.columns:
        open_ = out["Open"].astype(float)
        high = out["High"].astype(float)
        low = out["Low"].astype(float)
        close = out["Close"].astype(float)

        body = close - open_
        out["Body_Size"] = (body / open_.replace(0.0, np.nan)).fillna(0.0)
        upper_shadow = high - np.maximum(close, open_)
        lower_shadow = np.minimum(close, open_) - low
        out["Upper_Shadow"] = (
            upper_shadow / open_.replace(0.0, np.nan)
        ).fillna(0.0)
        out["Lower_Shadow"] = (
            lower_shadow / open_.replace(0.0, np.nan)
        ).fillna(0.0)
    else:
        out["Body_Size"] = 0.0
        out["Upper_Shadow"] = 0.0
        out["Lower_Shadow"] = 0.0

    # 12. Time features from index
    if isinstance(out.index, pd.DatetimeIndex):
        out["Day_Of_Week"] = out.index.dayofweek.astype(int)
        out["Day_Of_Month"] = out.index.day.astype(int)
    else:
        out["Day_Of_Week"] = 0
        out["Day_Of_Month"] = 0

    # 13. Lagged features and differences (memory)
    out["Close_Lag1"] = out["Close"].shift(1)
    out["Close_Diff"] = out["Close"].diff().fillna(0.0)

    if "RSI" in out.columns:
        out["RSI_Lag1"] = out["RSI"].shift(1)
        out["RSI_Diff"] = out["RSI"].diff().fillna(0.0)
    else:
        out["RSI_Lag1"] = np.nan
        out["RSI_Diff"] = 0.0

    out["Volume_Lag1"] = out["Volume"].shift(1)
    out["Volume_Diff"] = out["Volume"].diff().fillna(0.0)

    out["OBV_Lag1"] = out["OBV"].shift(1)
    out["OBV_Diff"] = out["OBV"].diff().fillna(0.0)

    # ---------------------------------------------------------
    # 14. Indicator Stacking (Signals + Rolling Win Rate)
    # ---------------------------------------------------------
    out = add_indicator_signals(out)
    out = add_rolling_win_rate(out, window=30)

    return out

def prepare_for_ai(df, target_pct: float = 0.03, stop_loss_pct: float = 0.06, look_forward_days: int = 20, use_volatility: bool = False):
    """
    Implements 'The Triple Barrier Method' labeling.
    
    IMPROVED: Now checks which barrier is hit FIRST (TP or SL).
    - Target = 1: Take Profit hit BEFORE Stop Loss
    - Target = 0: Stop Loss hit first OR neither hit (time barrier)
    """
    if df.empty: return df
    out = df.copy()
    
    # Entry Point (Next Day Close as entry)
    out['entry_price'] = out['Close'].shift(-1)
    
    out['tp_barrier'] = out['entry_price'] * (1 + target_pct)
    out['sl_barrier'] = out['entry_price'] * (1 - stop_loss_pct)
    
    # Initialize Target column
    out['Target'] = 0
    
    # For each row, we need to check which barrier is hit first
    # This requires iterating through future prices
    close_values = out['Close'].values
    tp_values = out['tp_barrier'].values
    sl_values = out['sl_barrier'].values
    targets = np.zeros(len(out), dtype=int)
    
    for i in range(len(out) - look_forward_days - 1):
        entry = close_values[i + 1]  # Entry at next day's close
        tp = tp_values[i]
        sl = sl_values[i]
        
        if np.isnan(entry) or np.isnan(tp) or np.isnan(sl):
            continue
            
        # Look at future prices (day i+1 to i+1+look_forward_days)
        first_tp_day = None
        first_sl_day = None
        
        for j in range(1, look_forward_days + 1):
            future_idx = i + 1 + j
            if future_idx >= len(close_values):
                break
                
            future_price = close_values[future_idx]
            if np.isnan(future_price):
                continue
            
            # Check if TP is hit (use future high if available, else close)
            if 'High' in out.columns:
                future_high = out['High'].values[future_idx]
            else:
                future_high = future_price
                
            if 'Low' in out.columns:
                future_low = out['Low'].values[future_idx]
            else:
                future_low = future_price
            
            # Record first day each barrier is hit
            if first_tp_day is None and future_high >= tp:
                first_tp_day = j
            if first_sl_day is None and future_low <= sl:
                first_sl_day = j
                
            # Early exit if both barriers are hit
            if first_tp_day is not None and first_sl_day is not None:
                break
        
        # Labeling Logic: TP hit FIRST = Win, otherwise = Loss
        if first_tp_day is not None:
            if first_sl_day is None or first_tp_day <= first_sl_day:
                targets[i] = 1  # TP hit first or SL never hit
            # else: SL hit first, target stays 0
        # else: Neither hit or only SL hit, target stays 0
    
    out['Target'] = targets
    
    # Stats for console monitoring
    counts = out['Target'].value_counts()
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    total = pos + neg
    ratio = pos / total if total > 0 else 0
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Triple Barrier Labeling: {len(out)} rows")
    print(f"    Wins={pos} ({ratio:.2%}), Losses={neg} ({1-ratio:.2%})")
    
    # Clean up and drop rows we can't label
    drop_cols = ['entry_price', 'tp_barrier', 'sl_barrier']
    out.drop(columns=drop_cols, inplace=True, errors='ignore')
    out = out.iloc[:-look_forward_days].copy()
    
    return out

# =============================================================================
# Training Monitor - Early Detection of Training Issues
# =============================================================================
class TrainingMonitor:
    """Monitor training progress and detect issues early."""
    
    def __init__(self, log_cb: Optional[Callable] = None):
        self.log_cb = log_cb
        self.alerts = []
    
    def _log(self, msg: str):
        """Log message via callback or print."""
        if self.log_cb:
            self.log_cb(msg)
        else:
            print(msg)
    
    def check_metrics(self, metrics: dict) -> list:
        """
        Check for problematic metric patterns.
        Returns list of alert messages.
        """
        self.alerts = []
        
        recall = metrics.get('recall', 0)
        precision = metrics.get('precision', 0)
        auc = metrics.get('auc', 1.0)
        
        # Alert 1: Perfect or near-perfect Recall (model predicting all 1s)
        if recall > 0.95:
            self.alerts.append(
                f"⚠️ CRITICAL: Recall={recall:.2%} - Model may be predicting BUY for everything!"
            )
        
        # Alert 2: Low AUC (barely better than random)
        if auc < 0.6:
            self.alerts.append(
                f"⚠️ WARNING: AUC={auc:.3f} - Model barely better than random (0.5)!"
            )
        
        # Alert 3: Large Precision-Recall gap
        if abs(precision - recall) > 0.3:
            self.alerts.append(
                f"⚠️ WARNING: Large P-R gap (P={precision:.2%} vs R={recall:.2%})"
            )
        
        # Alert 4: Very low precision
        if precision < 0.5:
            self.alerts.append(
                f"⚠️ WARNING: Precision={precision:.2%} - Too many false positives!"
            )
        
        return self.alerts
    
    def log_alerts(self):
        """Log all accumulated alerts."""
        if self.alerts:
            self._log("\n" + "="*60)
            self._log("⚠️ TRAINING ALERTS DETECTED:")
            for alert in self.alerts:
                self._log(f"  {alert}")
            self._log("="*60 + "\n")
    
    def check_class_balance(self, y) -> dict:
        """
        Check class balance and return statistics.
        """
        import numpy as np
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        stats = {}
        for cls, cnt in zip(unique, counts):
            stats[int(cls)] = {'count': int(cnt), 'pct': cnt / total}
        
        # Alert if heavily imbalanced
        if len(stats) == 2:
            pct_1 = stats.get(1, {}).get('pct', 0)
            if pct_1 > 0.8:
                self.alerts.append(
                    f"⚠️ IMBALANCE: {pct_1:.1%} of labels are positive (Target=1). "
                    "Consider adjusting target_pct/stop_loss_pct ratio."
                )
            elif pct_1 < 0.2:
                self.alerts.append(
                    f"⚠️ IMBALANCE: Only {pct_1:.1%} of labels are positive. "
                    "May need more data or different labeling strategy."
                )
        
        return stats


def calculate_optimal_class_weight(y, max_ratio: float = 5.0) -> dict:
    """
    Calculate class weights that prevent model from predicting all 1s.
    
    Args:
        y: Target labels (0 or 1)
        max_ratio: Maximum weight ratio to prevent extreme imbalance
        
    Returns:
        dict: Class weights {0: weight_0, 1: weight_1}
    """
    import numpy as np
    
    y_arr = np.array(y)
    pos = np.sum(y_arr == 1)
    neg = np.sum(y_arr == 0)
    total = pos + neg
    
    if pos == 0 or neg == 0:
        return {0: 1.0, 1: 1.0}
    
    # Calculate ratio
    # If more positives than negatives, weight negatives higher
    if pos > neg:
        ratio = min(pos / neg, max_ratio)
        weights = {0: ratio, 1: 1.0}
    else:
        ratio = min(neg / pos, max_ratio)
        weights = {0: 1.0, 1: ratio}
    
    print(f"[ClassWeight] Pos={pos} ({pos/total:.1%}), Neg={neg} ({neg/total:.1%})")
    print(f"[ClassWeight] Calculated weights: {weights}")
    
    return weights


class ModelTrainer:
    """
    Modular class for handling stock price data loading, 
    feature engineering, and model training.
    """
    def __init__(
        self, 
        exchange: str, 
        supabase_url: str, 
        supabase_key: str, 
        progress_cb: Optional[Callable[[Any], None]] = None
    ):
        self.exchange = self._standardize_exchange(exchange)
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.progress_cb = progress_cb
        self.market_df = None
        self.market_index_symbol = None
        self.market_index_loaded = False
        self.market_index_local_json = None
        self.fundamentals_loaded = False
        self.df_all = None
        self.predictors = []
        self.categorical_features = []
        self.min_history_needed = 200 # Default for safety (SMA200)
        self.embargo_pct = 0.01 # 1% embargo gap for purged k-fold

    def _clean_dataset(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Centralized cleaning to ensure X has correct dtypes for LightGBM.
        - Fills numeric NaNs with 0 (safe for trees).
        - Fills categorical NaNs with "Unknown".
        - Enforces 'category' dtype for self.categorical_features.
        """
        X = X.copy()
        
        # 1. Fill Numeric NaNs
        num_cols = X.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            X[num_cols] = X[num_cols].fillna(0)
            
        # 2. Handle Categorical Features
        for cat in self.categorical_features:
            if cat in X.columns:
                # Ensure it's not null before casting
                if X[cat].isnull().any():
                     X[cat] = X[cat].astype(object).fillna("Unknown")
                
                # Strict Cast to Category
                X[cat] = X[cat].astype('category')
                
        return X
        
    def _standardize_exchange(self, exchange: str) -> str:
        if not exchange: return "UNKNOWN"
        e_lower = exchange.strip().lower()
        if e_lower in ["ca", "cc", "cairo", "egypt"]:
            return "EGX"
        elif e_lower in ["us", "usa", "nasdaq", "nyse"]:
            return "US"
        else:
            return exchange.upper()

    def _progress(self, msg: str) -> None:
        print(msg)
        if self.progress_cb:
            try: self.progress_cb(msg)
            except Exception: pass

    def _progress_stats(self, phase: str, message: str, stats: Dict[str, Any]) -> None:
        if not self.progress_cb: return
        payload = {"phase": phase, "message": message, "stats": stats}
        try: self.progress_cb(payload)
        except Exception: pass

    def load_market_data(self) -> None:
        """Fetch Market Index Data for context."""
        candidates = []
        if self.exchange == "EGX": candidates = ["EGX30.INDX", "COMI.CA"]
        elif self.exchange == "US": candidates = ["GSPC.INDX", "SPY.US", "AAPL.US"]
        else: candidates = ["GSPC.INDX"]
            
        self.market_index_symbol = None
        self.market_index_loaded = False
        self.market_index_local_json = None
        try:
            if self.exchange == "EGX":
                api_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(api_dir)
                cand = os.path.join(project_root, "symbols_data", "EGX30-INDEX.json")
                if os.path.exists(cand):
                    self.market_index_local_json = os.path.join("symbols_data", "EGX30-INDEX.json")
        except Exception:
            pass
            
        for idx_sym in candidates:
            self._progress(f"Attempting to load market index data: {idx_sym}...")
            try:
                idx_res = (
                     self.supabase.table("stock_prices")
                    .select("date, close")
                    .eq("symbol", idx_sym)
                    .order("date", desc=False)
                    .execute()
                )
                if idx_res.data and len(idx_res.data) > 200:
                    df = pd.DataFrame(idx_res.data)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date").sort_index()
                    df['atr'] = df['close'].pct_change().rolling(20).std().fillna(0)
                    self.market_df = df
                    self.market_index_symbol = idx_sym
                    self.market_index_loaded = True
                    self._progress(f"Successfully loaded market context from {idx_sym}")
                    break
            except Exception as e:
                print(f"Warning: Failed to fetch market index {idx_sym}: {e}")

        # Fallback to local JSON if Database failed
        if self.market_df is None and self.market_index_local_json:
            try:
                api_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(api_dir)
                full_path = os.path.join(project_root, self.market_index_local_json)
                if os.path.exists(full_path):
                    import json
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        if data:
                            df = pd.DataFrame(data)
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.set_index("date").sort_index()
                            # Standardization: ensure 'close' column exists
                            if 'close' not in df.columns and 'Close' in df.columns:
                                df['close'] = df['Close']
                            
                            if 'close' in df.columns:
                                df['atr'] = df['close'].pct_change().rolling(20).std().fillna(0)
                                self.market_df = df
                                self.market_index_loaded = True
                                self._progress(f"Successfully loaded market context from local JSON: {self.market_index_local_json}")
            except Exception as e:
                self._progress(f"Warning: Failed to load local market index JSON: {e}")
                
        if self.market_df is None:
            self._progress("Warning: No market index data found. Market Context features will be 0.")

    def fetch_stock_prices(self, page_size: int = 1000) -> pd.DataFrame:
        """Fetch all stock prices for the exchange using parallel paging."""
        self._progress(f"Loading price data for exchange {self.exchange}...")
        
        # 1. Get total count
        rows_total = None
        try:
            count_res = self.supabase.table("stock_prices").select("symbol", count="exact")\
                .eq("exchange", self.exchange).limit(1).execute()
            rows_total = count_res.count
        except Exception as e:
            print(f"Warning: Failed to fetch total row count: {e}")

        # 2. Parallel Fetch
        def _fetch_page(off, retries=3):
            for attempt in range(retries):
                try:
                    res = self.supabase.table("stock_prices")\
                        .select("symbol, date, open, high, low, close, volume")\
                        .eq("exchange", self.exchange)\
                        .order("symbol", desc=False).order("date", desc=False)\
                        .range(off, off + page_size - 1).execute()
                    return res.data or []
                except Exception as e:
                    time.sleep((attempt + 1) * 2)
            return []

        all_rows = []
        first_page = _fetch_page(0)
        if not first_page: return pd.DataFrame()
        all_rows.extend(first_page)

        if rows_total and rows_total > page_size:
            offsets = range(page_size, rows_total, page_size)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(_fetch_page, o): o for o in offsets}
                for future in as_completed(futures):
                    data = future.result()
                    if data: all_rows.extend(data)
                    if len(all_rows) % (page_size * 10) == 0:
                        self._progress_stats("loading_rows", f"Loaded {len(all_rows):,} rows", {"rows_loaded": len(all_rows), "rows_total": rows_total})

        df = pd.DataFrame(all_rows)
        self._progress(f"Loaded {len(df):,} rows for {len(df['symbol'].unique()):,} symbols.")
        return df

    @staticmethod
    def _process_single_symbol(params):
        """Worker function for parallel processing. Must be static for pickleability."""
        try:
            sym, df_sym, market_df, target_pct, stop_loss_pct, look_forward_days, use_vol_label, min_history = params
            
            if len(df_sym) < min_history: return None

            # 1. Base Technical Indicators
            df = add_technical_indicators(df_sym)
            if df.empty: return None

            # Preserve fundamentals/categorical columns (merged into df_sym) through the feature pipeline.
            # add_technical_indicators() returns a new dataframe, so we must carry these columns forward.
            for _c in ("marketCap", "peRatio", "eps", "dividendYield", "sector", "industry"):
                if _c in df_sym.columns and _c not in df.columns:
                    try:
                        if len(df_sym[_c]) == len(df):
                            df[_c] = df_sym[_c].values
                        else:
                            df[_c] = df_sym[_c].iloc[-1]
                    except Exception:
                        df[_c] = df_sym[_c].iloc[-1] if _c in df_sym.columns and len(df_sym) else None
            
            # 2. Massive Feature Set
            df = add_massive_features(df)
            
            # 3. Market Context
            df = add_market_context(df, market_df)
            
            # 4. Labeling (The Triple Barrier Strategy)
            df = prepare_for_ai(df, target_pct, stop_loss_pct, look_forward_days, use_volatility=use_vol_label)
            
            # Require minimum history (redundant check but good for safety if indicators drop rows)
            if len(df) < 10: return None 
            df['symbol'] = sym
            return df
        except Exception as e:
            print(f"Error processing {params[0]}: {e}")
            return None

    def prepare_training_data(
        self, 
        df_all: pd.DataFrame, 
        target_pct: float, 
        stop_loss_pct: float, 
        look_forward_days: int,
        preset: str = "extended",
        use_volatility_label: bool = False,
    ) -> pd.DataFrame:
        """Process features in parallel for all symbols."""
        self._progress(f"Starting parallel feature engineering (Preset: {preset})...")
        
        # Determine min history based on preset
        self.min_history_needed = 200 if preset in ["extended", "max"] else 60
        
        # Merge fundamentals
        start_time = time.time()
        
        # Merge fundamentals
        df_funds = fetch_fundamentals_for_exchange(self.supabase, self.exchange)
        self.fundamentals_loaded = bool(df_funds is not None and (not df_funds.empty))
        if df_funds is not None and (not df_funds.empty):
            df_all = df_all.merge(df_funds, on="symbol", how="left")
            for cat_col in ["sector", "industry"]:
                if cat_col in df_all.columns:
                    df_all[cat_col] = df_all[cat_col].fillna("Unknown").astype("category")
                    if cat_col not in self.categorical_features:
                        self.categorical_features.append(cat_col)

        # Memory optimization
        df_all = _downcast_df(df_all)

        use_vol_label = bool(use_volatility_label)
        symbol_params = [
            (sym, df_sym, self.market_df, target_pct, stop_loss_pct, look_forward_days, use_vol_label, self.min_history_needed) 
            for sym, df_sym in df_all.groupby("symbol")
        ]
        
        combined_data = []
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            results = list(executor.map(self._process_single_symbol, symbol_params))
            combined_data = [res for res in results if res is not None]

        if not combined_data:
            raise ValueError("No valid data collected for training")

        df_train = pd.concat(combined_data)

        # Ensure categorical dtypes exist post-parallel concat (workers may coerce types)
        for cat_col in ["sector", "industry"]:
            if cat_col in df_train.columns:
                try:
                    df_train[cat_col] = df_train[cat_col].fillna("Unknown").astype("category")
                except Exception as e:
                    print(f"Warning: Failed to cast {cat_col} to category: {e}")
                    df_train[cat_col] = df_train[cat_col].fillna("Unknown")

        self._progress(f"Prepared training data: {len(df_train):,} total samples.")
        return df_train

    def select_predictors(self, df: pd.DataFrame, preset: str = "extended", max_features: Optional[int] = None):
        """Select feature set based on preset."""
        core = ["Close", "Volume", "SMA_50", "RSI", "MACD", "MACD_Signal", "MACD_Hist", "Z_Score"]
        extended = core + [
            "SMA_200", "EMA_50", "RSI_7", "BB_PctB", "BB_Width", "OBV", "OBV_Slope",
            "Dist_From_High", "Dist_From_Low", "Body_Size", "Upper_Shadow", "Lower_Shadow",
            "SMA_Cross", "EMA_Cross", "Price_vs_SMA200", 
            "Day_Of_Week", "Day_Of_Month", "Close_Lag1", "RSI_Lag1", "Volume_Lag1",
            "feat_rsi_signal", "feat_rsi_acc", "feat_ema_signal", "feat_ema_acc",
            "feat_bb_signal", "feat_bb_acc", "feat_mkt_trend", "feat_mkt_volatility", "feat_rel_strength"
        ]
        max_p = extended + ["ATR_14", "ADX_14", "STOCH_K", "STOCH_D", "CCI_20", "VWAP_20", "Momentum", "ROC_12", "VOL_SMA20", "VOL_Change"]
        
        self.predictors = []
        if preset == "max":
            # Dynamic Feature Selection: Use ALL numeric columns minus targets/metadata
            exclude = set(["Target", "Date", "Symbol", "Open", "High", "Low", "Close", "Volume"])
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.predictors = [c for c in numeric_cols if c not in exclude]
            # Ensure core features are kept even if logic above misses them (unlikely)
            for c in core:
                if c in df.columns and c not in self.predictors:
                    self.predictors.append(c)
        else:
            chosen = core if preset == "core" else extended
            self.predictors = [c for c in chosen if c in df.columns]
        
        # Add categorical if present
        for cf in self.categorical_features:
            if cf in df.columns and cf not in self.predictors:
                self.predictors.append(cf)

        # Keep categorical_features aligned with predictors/df columns to avoid KeyError during X = df[predictors]
        self.categorical_features = [c for c in self.categorical_features if c in df.columns and c in self.predictors]
            
        if max_features and max_features > 0:
            self.predictors = self.predictors[:max_features]
            
        self._progress(f"Selected {len(self.predictors)} predictors (Preset: {preset})")

    def optimize_hyperparameters(self, df_train: pd.DataFrame, n_trials: int = 75, patience: int = 50) -> Dict[str, Any]:
        """
        Use Optuna to find the best hyperparameters for LightGBM.
        Strategy: 'Safe for EGX'
        - Fixed Learning Rate: 0.01
        - Objective: Maximize (AUC + Precision) / 2
        - Constrained optimization space to prevent overfitting
        """
        if not optuna:
            self._progress("Optuna not installed. Skipping optimization.")
            return {}

        self._progress(f"Starting Hyperparameter Optimization with Optuna for EGX ({n_trials} trials)...")
        self._progress("Strategy: Fixed LR=0.01, Objective=(AUC+Precision)/2, Constrained Trees")
        
        X = df_train[self.predictors]
        # Clean data BEFORE splitting to ensure types are consistent
        X = self._clean_dataset(X)
        y = df_train["Target"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        def objective(trial):
            # 1. Constrained Search Space
            params = {
                "objective": "binary",
                "metric": "auc", # Monitor AUC during early stopping
                "verbosity": -1,
                "boosting_type": "gbdt",
                # Fixed Parameters
                "learning_rate": 0.01,
                # Optimized Parameters (Constrained)
                "n_estimators": trial.suggest_int("n_estimators", 300, 700),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                # Fixed structural params
                "num_leaves": trial.suggest_int("num_leaves", 20, 40), # Tied to max_depth roughly (2^depth)
            }

            model = lgb.LGBMClassifier(**params)
            
            # 2. Early Stopping (Only if Patience > 0)
            callbacks = []
            if patience and patience > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=patience))
                
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
            
            # 3. Custom Objective Function
            preds = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            # Safety check for single-class predictions
            if len(np.unique(preds)) < 2:
                # Penalize trivial solutions (all 0s or all 1s)
                return 0.0

            precision = precision_score(y_val, preds, zero_division=0)
            recall = recall_score(y_val, preds, zero_division=0)
            auc = roc_auc_score(y_val, y_prob)

            # Penalize cheating (Recall=1.0 often means it predicted all 1s)
            if recall > 0.99 or recall < 0.01:
                return 0.0
            
            # Target Metric: (AUC + Precision) / 2
            score = (auc + precision) / 2.0
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        self._progress(f"Optimization complete! Best Score: {study.best_value:.4f}")
        self._progress(f"Best Params: {study.best_params}")
        
        # Ensure fixed params are included in the return
        final_params = study.best_params.copy()
        final_params["learning_rate"] = 0.01
        
        return final_params

    def purged_cross_val(self, df: pd.DataFrame, n_splits: int = 5) -> List[Dict[str, float]]:
        """
        Implementation of Purged K-Fold Cross Validation.
        Ensures training data does not overlap (with embargo) with testing data.
        """
        if len(df) < (n_splits * 20): return []
        
        self._progress(f"Running Purged {n_splits}-Fold Cross Validation...")
        X = df[self.predictors]
        # Clean data to ensure categorical features are properly typed
        X = self._clean_dataset(X)
        y = df["Target"]
        # Use simple integer indexing for splitting
        indices = np.arange(len(df))
        test_size = len(df) // n_splits
        embargo_size = int(len(df) * self.embargo_pct)
        
        results = []
        for i in range(n_splits):
            test_start = i * test_size
            test_end = (i + 1) * test_size
            test_indices = indices[test_start:test_end]
            
            # Purging: ensure training doesn't leak from look-forward period
            # Embargo: extra buffer after the test set
            train_indices = np.concatenate([
                indices[:max(0, test_start - embargo_size)],
                indices[test_end + embargo_size:]
            ])
            
            if len(train_indices) < 50: continue
            
            X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
            X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
            
            # Simple model for CV
            model = LGBMClassifier(n_estimators=500, learning_rate=0.05, verbose=-1, is_unbalance=True)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            results.append({
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0)
            })
            
        return results

    def train_model(
        self, 
        df_train: pd.DataFrame, 
        n_estimators: int = 5000, 
        learning_rate: float = 0.05, 
        patience: int = 100,
        optimized_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Train LightGBM with early stopping and optional optimized params."""
        X = df_train[self.predictors]
        y = df_train["Target"]
        
        # Data Validation
        if X.isnull().values.any():
             self._progress("Warning: NaNs found in predictors. Applying centralized cleaning...")

        # Centralized Cleaning
        X = self._clean_dataset(X)
        self._progress(f"Data types before training: {X.dtypes.to_dict()}")

        if len(np.unique(y)) < 2:
            raise ValueError(f"Training failed: Only one class present in target ({np.unique(y)}). Need both Win and No-Win samples.")

        self._progress(f"Training LightGBM model (samples={len(X)}, target_pos={y.sum()})...")
        
        # Financial ML: Purged Time-Series Validation
        # Instead of random split or single split, we implement a simple embargo/purge logic
        # by ensuring a gap between training and validation data.
        test_size = 0.2
        split_idx = int(len(df_train) * (1 - test_size))
        
        # Embargo: remove look_forward_days from the end of training to avoid leakage
        # (Though we already dropped them in prepare_for_ai, this is extra safety)
        train_end_idx = max(0, split_idx - (patience + 20))
        
        X_train = X.iloc[:train_end_idx]
        y_train = y.iloc[:train_end_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        
        # Initialize Training Monitor for early issue detection
        monitor = TrainingMonitor(log_cb=self._progress)
        class_stats = monitor.check_class_balance(y_train)
        self._progress(f"Class balance: {class_stats}")
        
        # Optional: Run Purged CV for more reliable estimation
        avg_purged_f1 = None
        cv_scores = self.purged_cross_val(df_train, n_splits=3)
        if cv_scores:
            avg_purged_f1 = np.mean([s['f1'] for s in cv_scores])
            self._progress(f"Average Purged CV F1: {avg_purged_f1:.4f}")

        # Calculate optimal class weights instead of using is_unbalance=True
        # This gives more control and prevents model from predicting all 1s
        class_weight = calculate_optimal_class_weight(y_train, max_ratio=3.0)
        
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": 5,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "class_weight": class_weight,  # Use calculated weights instead of is_unbalance
            **(optimized_params or {})
        }
        
        model = LGBMClassifier(**params)
        

        callbacks = [
            lgb.log_evaluation(period=100),
            StreamCallback(self.progress_cb) if self.progress_cb else None
        ]
        
        if patience and patience > 0:
            callbacks.insert(0, lgb.early_stopping(stopping_rounds=patience))

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="logloss",
            # Use 'auto' - LightGBM will detect category dtype columns automatically
            # This avoids errors when column names are passed but dtype isn't category
            categorical_feature='auto',
            callbacks=[c for c in callbacks if c is not None]
        )
        
        # Evaluate
        metrics = self.calculate_validation_metrics(model, df_train)
        
        # Check for training issues and alert
        monitor.check_metrics(metrics)
        monitor.log_alerts()
        
        # Analyze and log feature importance
        self.analyze_feature_importance(model, top_n=20)
        
        return model, metrics, avg_purged_f1

    def calculate_validation_metrics(self, model, df_train: pd.DataFrame) -> Dict[str, float]:
        """Calculate final metrics on the validation split."""
        X = df_train[self.predictors]
        y = df_train["Target"]
        
        # Consistent with train_model's split
        test_size = 0.2
        split_idx = int(len(df_train) * (1 - test_size))
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        
        if len(np.unique(y_val)) < 2:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.5}

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        metrics = {
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_val, y_prob))
        }
        self._progress(f"Final Validation Metrics: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        return metrics

    def analyze_feature_importance(
        self, 
        model, 
        top_n: int = 30,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze and log feature importance after training.
        
        Args:
            model: Trained LightGBM model
            top_n: Number of top features to display
            save_path: Optional path to save CSV
            
        Returns:
            DataFrame with feature importance
        """
        try:
            importance = model.feature_importances_
            
            feat_imp = pd.DataFrame({
                'feature': self.predictors,
                'importance': importance,
                'importance_pct': (importance / importance.sum()) * 100
            }).sort_values('importance', ascending=False)
            
            # Log top features
            self._progress(f"\n📊 Top {top_n} Most Important Features:")
            for idx, row in feat_imp.head(top_n).iterrows():
                bar = "█" * int(row['importance_pct'])
                self._progress(f"  {row['feature']}: {row['importance_pct']:.2f}% {bar}")
            
            # Identify low-importance features (< 0.1%)
            low_imp = feat_imp[feat_imp['importance_pct'] < 0.1]
            if len(low_imp) > 0:
                self._progress(f"\n⚠️ {len(low_imp)} features with <0.1% importance (consider removing)")
            
            # Save to file if path provided
            if save_path:
                feat_imp.to_csv(save_path, index=False)
                self._progress(f"Feature importance saved to {save_path}")
            
            return feat_imp
            
        except Exception as e:
            self._progress(f"Warning: Could not analyze feature importance: {e}")
            return pd.DataFrame()

    def save_model(self, model, filename: str, metadata: Dict[str, Any]) -> str:
        """Save model and metadata locally and to cloud."""
        api_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(api_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        filepath = os.path.join(models_dir, filename)
        booster = getattr(model, "booster_", None)
        
        num_features = None
        num_trees = None
        try:
            if booster is not None:
                num_features = booster.num_feature()
                num_trees = booster.num_trees()
        except Exception:
            pass

        training_samples = metadata.get("trainingSamples")
        if training_samples is None:
            training_samples = metadata.get("training_samples")

        feature_preset = metadata.get("featurePreset")
        if feature_preset is None:
            feature_preset = metadata.get("feature_preset")

        artifact = {
            "kind": "lgbm_booster",
            "model_str": booster.model_to_string() if booster else None,
            "feature_names": self.predictors,
            "categorical_features": self.categorical_features,
            "exchange": self.exchange,
            "featurePreset": feature_preset,
            "trainingSamples": training_samples,
            "num_features": num_features,
            "num_trees": num_trees,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **metadata
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(artifact if booster else model, f)
            
        try:
            uses_fundamentals = False
            for c in ("marketCap", "peRatio", "eps", "dividendYield", "sector", "industry"):
                if c in self.predictors:
                    uses_fundamentals = True
                    break

            uses_exchange_index_json = bool(self.market_index_local_json)
            has_meta_labeling = False
            if isinstance(metadata, dict):
                has_meta_labeling = bool(metadata.get("has_meta_labeling") or metadata.get("meta_labeling"))

            card = {
                "model_name": filename,
                "created_at": artifact.get("timestamp"),
                "exchange": self.exchange,
                "artifact_kind": artifact.get("kind"),
                "feature_preset": feature_preset,
                "training": {
                    "training_samples": training_samples,
                    "target_pct": metadata.get("target_pct"),
                    "stop_loss_pct": metadata.get("stop_loss_pct"),
                    "look_forward_days": metadata.get("look_forward_days"),
                    "learning_rate": metadata.get("learning_rate"),
                    "metrics": {
                        "precision": metadata.get("precision"),
                        "recall": metadata.get("recall"),
                        "f1": metadata.get("f1"),
                        "auc": metadata.get("auc"),
                    },
                },
                "data_inputs": {
                    "uses_exchange_index_json": uses_exchange_index_json,
                    "exchange_index_json_path": self.market_index_local_json,
                    "uses_exchange_index_data": bool(self.market_index_loaded),
                    "exchange_index_symbol": self.market_index_symbol,
                    "uses_fundamentals": uses_fundamentals,
                    "fundamentals_loaded": bool(self.fundamentals_loaded),
                },
                "capabilities": {
                    "has_meta_labeling": has_meta_labeling,
                },
            }

            card_path = os.path.join(models_dir, f"{filename}.model_card.json")
            with open(card_path, "w", encoding="utf-8") as cf:
                json.dump(card, cf)
        except Exception as e:
            self._progress(f"Warning: failed to write model card for {filename}: {e}")

        self._progress(f"Model saved: {filepath}")
        return filepath

    def save_meta_labeling_system(
        self,
        primary_model,
        meta_model,
        filename: str,
        metadata: Dict[str, Any],
        meta_threshold: float,
        meta_feature_names: List[str],
    ) -> str:
        api_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(api_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        filepath = os.path.join(models_dir, filename)
        booster = getattr(primary_model, "booster_", None)

        primary_artifact = {
            "kind": "lgbm_booster",
            "model_str": booster.model_to_string() if booster else None,
            "feature_names": self.predictors,
            "categorical_features": self.categorical_features,
            "exchange": self.exchange,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **metadata,
        }

        artifact = {
            "kind": "meta_labeling_system",
            "primary_model": primary_artifact,
            "meta_model": meta_model,
            "meta_feature_names": list(meta_feature_names or []),
            "meta_threshold": float(meta_threshold),
            "exchange": self.exchange,
            "timestamp": primary_artifact.get("timestamp"),
            **metadata,
        }

        with open(filepath, "wb") as f:
            pickle.dump(artifact, f)

        try:
            uses_fundamentals = False
            for c in ("marketCap", "peRatio", "eps", "dividendYield", "sector", "industry"):
                if c in self.predictors:
                    uses_fundamentals = True
                    break

            uses_exchange_index_json = bool(self.market_index_local_json)

            training_samples = metadata.get("trainingSamples")
            if training_samples is None:
                training_samples = metadata.get("training_samples")

            feature_preset = metadata.get("featurePreset")
            if feature_preset is None:
                feature_preset = metadata.get("feature_preset")

            card = {
                "model_name": filename,
                "created_at": artifact.get("timestamp"),
                "exchange": self.exchange,
                "artifact_kind": artifact.get("kind"),
                "feature_preset": feature_preset,
                "training": {
                    "training_samples": training_samples,
                    "target_pct": metadata.get("target_pct"),
                    "stop_loss_pct": metadata.get("stop_loss_pct"),
                    "look_forward_days": metadata.get("look_forward_days"),
                    "learning_rate": metadata.get("learning_rate"),
                    "metrics": {
                        "precision": metadata.get("precision"),
                        "recall": metadata.get("recall"),
                        "f1": metadata.get("f1"),
                        "auc": metadata.get("auc"),
                    },
                },
                "data_inputs": {
                    "uses_exchange_index_json": uses_exchange_index_json,
                    "exchange_index_json_path": self.market_index_local_json,
                    "uses_exchange_index_data": bool(self.market_index_loaded),
                    "exchange_index_symbol": self.market_index_symbol,
                    "uses_fundamentals": uses_fundamentals,
                    "fundamentals_loaded": bool(self.fundamentals_loaded),
                },
                "capabilities": {
                    "has_meta_labeling": True,
                    "meta_threshold": float(meta_threshold),
                },
            }

            card_path = os.path.join(models_dir, f"{filename}.model_card.json")
            with open(card_path, "w", encoding="utf-8") as cf:
                json.dump(card, cf)
        except Exception as e:
            self._progress(f"Warning: failed to write model card for {filename}: {e}")

        self._progress(f"Model saved: {filepath}")
        return filepath

def train_model(exchange=None, supabase_url=None, supabase_key=None, *args, **kwargs):
    """Wrapper for backward compatibility and CLI."""
    if exchange is None:
        exchange = kwargs.get("exchange")
    if supabase_url is None:
        supabase_url = kwargs.get("supabase_url")
    if supabase_key is None:
        supabase_key = kwargs.get("supabase_key")

    trainer = ModelTrainer(
        exchange=exchange,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        progress_cb=kwargs.get("progress_cb")
    )
    
    trainer.load_market_data()
    df_raw = trainer.fetch_stock_prices()
    if df_raw.empty: return
    
    use_volatility_label = bool(kwargs.get("use_volatility_label", False))
    # Use kwargs for labeling parameters (important for realistic training)
    target_pct = kwargs.get("target_pct", 0.03)
    stop_loss_pct = kwargs.get("stop_loss_pct", 0.06)
    look_forward_days = kwargs.get("look_forward_days", 20)
    
    df_train = trainer.prepare_training_data(
        df_raw, 
        target_pct, 
        stop_loss_pct, 
        look_forward_days,
        preset=kwargs.get("feature_preset", "extended"),
        use_volatility_label=use_volatility_label,
    )
    
    max_features = kwargs.get("max_features")
    if max_features is None:
        max_features = kwargs.get("max_features_override")
    trainer.select_predictors(df_train, kwargs.get("feature_preset", "extended"), max_features)
    
    optimized_params = None
    if kwargs.get("optimize") and optuna:
        optimized_params = trainer.optimize_hyperparameters(df_train, n_trials=kwargs.get("n_trials", 30))

    # Get use_early_stopping from kwargs (defaults to True for backward compat)
    use_early_stopping = kwargs.get("use_early_stopping", True)
    
    model, val_metrics, avg_purged_f1 = trainer.train_model(
        df_train, 
        n_estimators=kwargs.get("n_estimators") or 500,
        learning_rate=kwargs.get("learning_rate", 0.01),
        patience=kwargs.get("patience", 100) if use_early_stopping else 0,
        optimized_params=optimized_params
    )
    
    # Save
    filename = kwargs.get("model_name") or f"model_{trainer.exchange}.pkl"
    if not filename.endswith(".pkl"): filename += ".pkl"
    
    use_meta_labeling = bool(kwargs.get("use_meta_labeling", True))
    meta_threshold = float(kwargs.get("meta_threshold", 0.3))

    metadata = {
        "target_pct": kwargs.get("target_pct"),
        "stop_loss_pct": kwargs.get("stop_loss_pct"),
        "look_forward_days": kwargs.get("look_forward_days"),
        "use_volatility_label": use_volatility_label,
        "feature_preset": kwargs.get("feature_preset"),
        "featurePreset": kwargs.get("feature_preset"),
        "n_estimators": getattr(model, "best_iteration_", kwargs.get("n_estimators")),
        "learning_rate": kwargs.get("learning_rate"),
        "training_samples": len(df_train),
        "trainingSamples": len(df_train),
        "optimized": optimized_params is not None,
        "purged_cv_f1": avg_purged_f1,
        "has_meta_labeling": bool(use_meta_labeling),
        "meta_threshold": float(meta_threshold),
        **val_metrics
    }
    if optimized_params:
        metadata["optuna_params"] = optimized_params
    
    if use_meta_labeling:
        if xgb is None:
            trainer._progress("Warning: xgboost not installed; falling back to LightGBM for meta-labeling.")

        X_primary = df_train[trainer.predictors].copy()
        y_primary = df_train["Target"].astype(int).copy()

        try:
            primary_probs = model.predict_proba(X_primary)[:, 1]
            primary_preds = (primary_probs >= 0.5).astype(int)
        except Exception:
            primary_preds = model.predict(X_primary)
            try:
                primary_probs = model.predict_proba(X_primary)[:, 1]
            except Exception:
                primary_probs = primary_preds.astype(float)

        meta_feature_names = []
        for c in trainer.predictors:
            if c in (trainer.categorical_features or []):
                continue
            try:
                if is_numeric_dtype(df_train[c]):
                    meta_feature_names.append(c)
            except Exception:
                continue
        X_meta_base = df_train[meta_feature_names].copy()
        X_meta_base = X_meta_base.replace([np.inf, -np.inf], np.nan).fillna(0)

        mask = (primary_preds == 1)
        if int(mask.sum()) < 50:
            mask = np.ones(len(df_train), dtype=bool)

        X_meta = X_meta_base.loc[mask].copy()
        X_meta["primary_prob"] = np.asarray(primary_probs)[mask]
        y_meta = y_primary.loc[mask].values

        if len(np.unique(y_meta)) < 2:
            raise ValueError("Meta-labeling training failed: only one class present for meta labels")

        pos = float(np.sum(y_meta == 1))
        neg = float(np.sum(y_meta == 0))
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        if xgb is not None:
            meta_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            )
        else:
            # LightGBM fallback (keeps meta-labeling available without xgboost)
            meta_model = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
        meta_model.fit(X_meta, y_meta)

        meta_feature_names_with_prob = list(meta_feature_names) + ["primary_prob"]
        trainer.save_meta_labeling_system(
            primary_model=model,
            meta_model=meta_model,
            filename=filename,
            metadata=metadata,
            meta_threshold=meta_threshold,
            meta_feature_names=meta_feature_names_with_prob,
        )
    else:
        metadata["has_meta_labeling"] = False
        trainer.save_model(model, filename, metadata)
    
    # Update global summary for the dashboard
    summary = {
        "exchange": trainer.exchange,
        "model_name": filename,
        "timestamp": datetime.now().isoformat(),
        "metrics": val_metrics,
        "features_count": len(trainer.predictors),
        "samples": len(df_train),
        "has_meta_labeling": bool(use_meta_labeling),
        "meta_threshold": float(meta_threshold),
    }
    _write_training_summary(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--optimize", action="store_true", help="Use Optuna to tune hyperparameters")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    args = parser.parse_args()
    
    train_model(
        exchange=args.exchange,
        supabase_url=os.getenv("NEXT_PUBLIC_SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        learning_rate=args.learning_rate,
        optimize=args.optimize,
        n_trials=args.trials
    )
