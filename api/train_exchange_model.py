import os
import warnings
import time

# Suppress specific FutureWarnings from libraries like 'ta'
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import argparse
import pickle
import json
from typing import Optional, Callable, Any, Dict

import pandas as pd
import numpy as np
from datetime import datetime
from ta import add_all_ta_features
from lightgbm import LGBMClassifier
import lightgbm as lgb
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, make_scorer
from supabase import create_client, Client
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for potential imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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



def optimize_and_train_model(X_train, y_train):
    scorer = make_scorer(precision_score)
    base_model = LGBMClassifier(random_state=1, n_jobs=-1, verbose=-1)
    param_grid = {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 50],
        "max_depth": [-1, 10, 20],
    }
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("Best parameters (precision):", search.best_params_)
    print("Best precision score:", search.best_score_)
    return search.best_estimator_

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

def prepare_for_ai(df, target_pct: float = 0.15, stop_loss_pct: float = 0.05, look_forward_days: int = 20):
    """
    Implements 'The Sniper Strategy' labeling with configurable parameters.
    - Target: +target_pct gain within look_forward_days.
    - Stop Loss: -stop_loss_pct loss (must not be hit before the target).
    """
    if df.empty: return df
    out = df.copy()
    
    # Use FixedForwardWindowIndexer for forward-looking rolling windows
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=look_forward_days)
    
    # Fix Gap Trap: Entry is Next Day Open
    out['next_open'] = out['Open'].shift(-1)
    
    # Future High/Low: Start checking from next day (shift -1)
    # This aligns the window [t+1, t+1+look_forward] to row t
    out['future_high'] = out['High'].shift(-1).rolling(window=indexer).max()
    out['future_low'] = out['Low'].shift(-1).rolling(window=indexer).min()
    
    # Labeling Logic
    out['Target'] = 0
    
    # Condition: Future High >= Entry + Target
    hit_target = out['future_high'] >= (out['next_open'] * (1 + target_pct))
    
    # Condition: Future Low > Entry - Stop (Stop NOT hit)
    safe_from_stop = out['future_low'] > (out['next_open'] * (1 - stop_loss_pct))
    
    # Mark as Buy (1) only if both conditions met
    out.loc[hit_target & safe_from_stop, 'Target'] = 1
    
    # Remove rows where we can't look forward (the last 'look_forward_days' rows)
    # and helper columns
    out.drop(columns=['future_high', 'future_low', 'next_open'], inplace=True, errors='ignore')
    out = out.iloc[:-look_forward_days].copy()
    
    return out

def train_model(
    exchange: str,
    supabase_url: str,
    supabase_key: str,
    use_early_stopping: bool = True,
    n_estimators: Optional[int] = None,
    model_name: Optional[str] = None,
    upload_to_cloud: bool = True,
    feature_preset: str = "extended",  # "core" | "extended" | "max"
    max_features_override: Optional[int] = None,
    training_strategy: Optional[str] = None,
    random_search_iter: Optional[int] = None,
    max_features: Optional[int] = None,
    progress_cb: Optional[Callable[[Any], None]] = None,
    target_pct: float = 0.15,
    stop_loss_pct: float = 0.05,
    look_forward_days: int = 20,
    use_grid: bool = False,
    learning_rate: float = 0.05,
    patience: int = 50,
):
    # Standardize exchange
    if exchange:
        e_lower = exchange.strip().lower()
        if e_lower in ["ca", "cc", "cairo", "egypt"]:
            exchange = "EGX"
        elif e_lower in ["us", "usa", "nasdaq", "nyse"]:
            exchange = "US"
        else:
            exchange = exchange.upper()

    print(f"Starting training for exchange: {exchange} (Grid Search: {use_grid})")
    start_time_total = time.time()
    supabase: Client = create_client(supabase_url, supabase_key)
    def _progress(msg: str) -> None:
        print(msg)
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass
    def _progress_stats(phase: str, message: str, stats: Dict[str, Any]) -> None:
        if not progress_cb:
            return
        payload = {
            "phase": phase,
            "message": message,
            "stats": stats,
        }
        try:
            progress_cb(payload)
        except Exception:
            pass
    
    # 1. Bulk fetch all price data for this exchange
    _progress(f"Loading price data for exchange {exchange}...")
    start_time_loading = time.time()
    
    # NEW: Fetch Market Index Data first
    # Strategy: 
    # 1. Try "EGX30.INDX" (standard index)
    # 2. Fallback to "COMI.CA" (Blue chip proxy, often more reliable data availability)
    
    market_df = None
    
    if exchange == "EGX":
        candidates = ["EGX30.INDX", "COMI.CA"]
    elif exchange == "US":
        candidates = ["GSPC.INDX", "SPY.US", "AAPL.US"]
    else:
        candidates = ["GSPC.INDX"] # Default global proxy
        
    for idx_sym in candidates:
        _progress(f"Attempting to load market index data: {idx_sym}...")
        try:
            idx_res = (
                 supabase.table("stock_prices")
                .select("date, close") # We only need close for now (and date)
                .eq("symbol", idx_sym)
                .order("date", desc=False)
                .execute()
            )
            if idx_res.data and len(idx_res.data) > 200:
                market_df = pd.DataFrame(idx_res.data)
                market_df["date"] = pd.to_datetime(market_df["date"])
                market_df = market_df.set_index("date").sort_index()
                # Calculate simple volatility proxy for market
                market_df['atr'] = market_df['close'].pct_change().rolling(20).std().fillna(0)
                _progress(f"Successfully loaded market context from {idx_sym}")
                break
        except Exception as e:
            print(f"Warning: Failed to fetch market index {idx_sym}: {e}")
            
    if market_df is None:
        _progress("Warning: No market index data found. Market Context features will be 0.")

    rows_total = None
    try:
        count_res = (
            supabase.table("stock_prices")
            .select("symbol", count="exact")
            .eq("exchange", exchange)
            .limit(1)
            .execute()
        )
        rows_total = count_res.count
    except Exception as e:
        print(f"Warning: Failed to fetch total row count: {e}")
        rows_total = None

    all_rows = []
    page_size = 1000
    
    _progress(f"Starting parallel data load for {exchange}...")
    
    def _fetch_page(off, retries=3):
        for attempt in range(retries):
            try:
                # Optimized query: select only needed columns, use PK-compatible order
                res = (
                    supabase.table("stock_prices")
                    .select("symbol, date, open, high, low, close, volume")
                    .eq("exchange", exchange)
                    .order("symbol", desc=False)
                    .order("date", desc=False)
                    .range(off, off + page_size - 1)
                    .execute()
                )
                return res.data or []
            except Exception as e:
                wait = (attempt + 1) * 2
                print(f"Fetch error at offset {off}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        return []

    # Initial fetch to get first page and confirm data exists
    first_page = _fetch_page(0)
    if not first_page:
        _progress(f"No price data found or first fetch failed for exchange {exchange}")
        return
    
    all_rows.extend(first_page)
    
    # If we have rows_total, calculate remaining offsets
    if rows_total and rows_total > page_size:
        offsets = range(page_size, rows_total, page_size)
        
        # Parallel fetch remaining pages
        max_fetch_workers = 5 # Conservative parallel workers
        with ThreadPoolExecutor(max_workers=max_fetch_workers) as executor:
            future_to_offset = {executor.submit(_fetch_page, o): o for o in offsets}
            
            for future in as_completed(future_to_offset):
                offset = future_to_offset[future]
                try:
                    data = future.result()
                    if data:
                        all_rows.extend(data)
                        
                    # Periodic progress update
                    if len(all_rows) % (page_size * 5) == 0 or len(all_rows) >= (rows_total or 0):
                        msg = f"Loaded {len(all_rows):,} rows so far"
                        _progress_stats(
                            "loading_rows",
                            msg,
                            {"rows_loaded": len(all_rows), "rows_total": rows_total},
                        )
                except Exception as e:
                    print(f"Error processing page at offset {offset}: {e}")
    else:
        # Fallback for if rows_total is unknown: sequential fetch until empty
        # This is rare as the initial fetch usually gives us info or we already have it
        if rows_total is None:
            offset = page_size
            while True:
                batch = _fetch_page(offset)
                if not batch:
                    break
                all_rows.extend(batch)
                msg = f"Loaded {len(all_rows):,} rows (sequential fallback)"
                _progress_stats("loading_rows", msg, {"rows_loaded": len(all_rows)})
                if len(batch) < page_size:
                    break
                offset += page_size

    if not all_rows:
        _progress(f"No price data found for exchange {exchange}")
        return

    df_all = pd.DataFrame(all_rows)
    if df_all.empty:
        _progress(f"No price data found for exchange {exchange}")
        return

    duration_loading = time.time() - start_time_loading
    print(f"DEBUG: Data loading took {duration_loading:.2f}s for {len(all_rows)} rows")

    all_symbols = sorted(df_all["symbol"].dropna().unique().tolist())
    raw_rows = len(df_all)
    msg = f"Loaded {raw_rows:,} rows for {len(all_symbols):,} symbols ({exchange})"
    _progress(msg)
    _progress_stats(
        "data_loaded",
        msg,
        {"raw_rows": raw_rows, "symbols_total": len(all_symbols), "rows_total": raw_rows},
    )

    # --- Fetch and merge fundamentals ---
    df_funds = fetch_fundamentals_for_exchange(supabase, exchange)
    categorical_cols = []
    if not df_funds.empty:
        _progress(f"Loaded fundamentals for {len(df_funds)} symbols.")
        df_all = df_all.merge(df_funds, on="symbol", how="left")
        # Pre-process categorical features (sector, industry) for LGBM
        # Fill NA with "Unknown" BEFORE converting to category to avoid setitem error
        for cat_col in ["sector", "industry"]:
            if cat_col in df_all.columns:
                df_all[cat_col] = df_all[cat_col].fillna("Unknown").astype("category")
                categorical_cols.append(cat_col)
    else:
        _progress("No fundamentals found for this exchange. Proceeding with price data only.")

    combined_data = []
    symbols_used = 0
    symbols_total = len(all_symbols)
    symbols_processed = 0
    progress_step = max(1, symbols_total // 50) if symbols_total else 1
    
    start_time_features = time.time()
    
    _progress(f"Starting parallel feature engineering for {symbols_total} symbols...")
    
    # Prepare arguments for parallel processing - include context
    symbol_params = [
        (sym, df_sym, market_df, target_pct, stop_loss_pct, look_forward_days)
        for sym, df_sym in df_all.groupby("symbol")
    ]
    
    # Process in parallel using ProcessPool for CPU-bound TA calculations
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        # Use map to process all symbols
        results = list(executor.map(_process_single_symbol, symbol_params))
        
        # Collect valid results
        for i, res in enumerate(results):
            symbols_processed += 1
            if res is not None:
                combined_data.append(res)
                symbols_used += 1
                
            # Progress update (batched)
            if symbols_processed % progress_step == 0 or symbols_processed == symbols_total:
                msg = f"Processed {symbols_processed}/{symbols_total} symbols"
                # Only log every 5th step to reduce spam, or if it's the last one
                if symbols_processed % (progress_step * 5) == 0 or symbols_processed == symbols_total:
                    _progress(msg)
                    _progress_stats(
                        "processing_symbols",
                        msg,
                        {"symbols_processed": symbols_processed, "symbols_total": symbols_total},
                    )

    if not combined_data:
        _progress("No valid data collected for training")
        return

    duration_features = time.time() - start_time_features
    print(f"DEBUG: Feature engineering took {duration_features:.2f}s ({duration_features/symbols_total:.3f}s/symbol avg)")

    df_train = pd.concat(combined_data)
    total_samples = len(df_train)
    msg = f"Prepared training data: {symbols_used:,} symbols used, {total_samples:,} samples"
    _progress(msg)
    _progress_stats(
        "data_prepared",
        msg,
        {"symbols_used": symbols_used, "samples": total_samples},
    )

    # 3. Select feature set based on preset
    core_predictors = [
        "Close",
        "Volume",
        "SMA_50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "Z_Score",
    ]

    extended_predictors = [
        # Core price/volume and trend
        "Close",
        "Volume",
        "SMA_50",
        "SMA_200",
        "EMA_50",
        "MACD",
        "MACD_Signal",
        "RSI",

        # Bollinger and volatility context
        "BB_PctB",
        "BB_Width",

        # Volume structure
        "OBV",
        "OBV_Slope",

        # Distance from recent extremes
        "Dist_From_High",
        "Dist_From_Low",

        # Standardization and candle geometry
        "Z_Score",
        "Body_Size",
        "Upper_Shadow",
        "Lower_Shadow",

        # Calendar features
        "Day_Of_Week",
        "Day_Of_Month",

        # Lagged values and differences (memory)
        "Close_Lag1",
        "Close_Diff",
        "RSI_Lag1",
        "RSI_Diff",
        "Volume_Lag1",
        "Volume_Diff",
        "OBV_Lag1",
        "OBV_Diff",

        # Smart Indicators (Stacking)
        "feat_rsi_signal", "feat_rsi_acc",
        "feat_ema_signal", "feat_ema_acc",
        "feat_bb_signal", "feat_bb_acc",

        # Market Context
        "feat_mkt_trend", "feat_mkt_volatility", "feat_rel_strength",
    ]

    max_predictors = extended_predictors + [
        # Extra volatility/advanced indicators already computed above
        "ATR_14",
        "ADX_14",
        "STOCH_K",
        "STOCH_D",
        "CCI_20",
        "VWAP_20",
        # Extra momentum/volume context
        "Momentum",
        "ROC_12",
        "VOL_SMA20",
    ]

    preset = (feature_preset or "extended").strip().lower()
    if preset == "core":
        chosen = core_predictors
    elif preset == "max":
        chosen = max_predictors
    else:
        chosen = extended_predictors

    # Ensure all chosen columns exist (in case of legacy data)
    predictors = [c for c in chosen if c in df_train.columns]
    
    # Add categorical fundamentals if they exist
    categorical_features = [c for c in ["sector", "industry"] if c in df_train.columns]
    for cf in categorical_features:
        if cf not in predictors:
            predictors.append(cf)

    if preset == "max":
        exclude = {"Target", "symbol", "date", "datetime", "timestamp"}
        all_numeric = [
            c
            for c in df_train.columns
            if c not in exclude and c not in predictors and is_numeric_dtype(df_train[c])
        ]
        predictors = predictors + all_numeric
    
    if max_features_override is None and max_features is not None:
        max_features_override = max_features

    # Apply max_features_override if provided and positive
    if max_features_override is not None and max_features_override > 0:
        predictors = predictors[:max_features_override]

    msg = f"Feature preset '{preset}' -> {len(predictors)} predictors"
    _progress(msg)
    _progress_stats(
        "features_ready",
        msg,
        {"features": len(predictors), "feature_preset": preset},
    )

    X = df_train[predictors]
    y = df_train["Target"]

    start_time_train = time.time()

    # Use a large number of estimators with early stopping to prevent overfitting.
    if use_grid:
        msg = f"Starting GridSearchCV optimization..."
        _progress(msg)
        _progress_stats("training", msg, {"grid_search": True})
        
        # optimize_and_train_model already prints best params
        model = optimize_and_train_model(X, y)
        final_n_estimators = getattr(model, "n_estimators", 100)
    elif use_early_stopping:
        final_n_estimators = n_estimators or 5000  # Increased for better convergence
        msg = f"Training LightGBM (early_stopping=True, n_estimators={final_n_estimators})"
        _progress(msg)
        _progress_stats(
            "training",
            msg,
            {"n_estimators": final_n_estimators, "early_stopping": True},
        )
        # Split data (preserve time order for stock data)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            shuffle=False,  # Critical for time-series data
        )
        # Debug: log final n_estimators for early-stopping mode
        print(f"Final n_estimators: {final_n_estimators}, Training samples: {len(X_train)}, Validation: {len(X_val)}")
        model = LGBMClassifier(
            n_estimators=final_n_estimators,
            learning_rate=learning_rate,  # User configured
            max_depth=10,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="logloss",
            categorical_feature=categorical_features if categorical_features else 'auto',
            callbacks=[
                lgb.early_stopping(stopping_rounds=patience),
                lgb.log_evaluation(period=100),  # Log every 100 iterations
                StreamCallback(progress_cb) if progress_cb else None, # Real-time frontend updates
            ],
        )
    else:
        # Plain training without early stopping, n_estimators fully controlled by caller.
        final_n_estimators = n_estimators or 200
        msg = f"Training LightGBM (early_stopping=False, n_estimators={final_n_estimators})"
        _progress(msg)
        _progress_stats(
            "training",
            msg,
            {"n_estimators": final_n_estimators, "early_stopping": False},
        )
        # Debug: log final n_estimators for non-early-stopping mode
        print("Final n_estimators:", final_n_estimators)
        model = LGBMClassifier(
            n_estimators=final_n_estimators,
            random_state=1,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X, y)
    
    duration_train = time.time() - start_time_train
    print(f"DEBUG: Training took {duration_train:.2f}s")
    
    total_duration = time.time() - start_time_total
    print(f"DEBUG: Total pipeline took {total_duration:.2f}s")
    
    # Persist a lightweight summary for the UI
    try:
        num_features = int(getattr(model, "n_features_in_", X.shape[1]))
        if num_features <= 20:
            print(f"Warning: model for {exchange} has only {num_features} features (expected > 20)")
        summary = {
            "exchange": exchange,
            "useEarlyStopping": use_early_stopping,
            "nEstimators": int(final_n_estimators),
            "bestIteration": int(getattr(model, "best_iteration_", 0)) if use_early_stopping else None,
            "trainingSamples": int(total_samples),
            "numFeatures": num_features,
            "featurePreset": preset,
            "symbolsUsed": int(symbols_used),
            "rawRows": int(raw_rows),
            "targetPct": target_pct,
            "stopLossPct": stop_loss_pct,
            "lookForwardDays": look_forward_days,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "timing": {
                "loading": duration_loading,
                "features": duration_features,
                "training": duration_train,
                "total": total_duration
            }
        }
        _write_training_summary(summary)
    except Exception as e:
        print(f"Failed to write training summary JSON: {e}")
    
    # 4. Save and Upload
    api_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(api_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Resolve filename: either user-provided or default
    if model_name:
        safe_name = os.path.basename(model_name.strip())
        if not safe_name.endswith(".pkl"):
            safe_name = f"{safe_name}.pkl"
        filename = safe_name
    else:
        filename = f"model_{exchange}.pkl"

    filepath = os.path.join(models_dir, filename)
    
    booster = getattr(model, "booster_", None)
    if booster is not None:
        artifact = {
            "kind": "lgbm_booster",
            "model_str": booster.model_to_string(),
            "feature_names": predictors,
            "n_estimators": int(final_n_estimators),
            "num_features": int(len(predictors)),
            "num_trees": int(getattr(booster, "num_trees", lambda: 0)()),
            "exchange": exchange,
            "featurePreset": preset,
            "trainingSamples": int(total_samples),
            "bestIteration": int(getattr(model, "best_iteration_", 0)) if use_early_stopping else None,
            "target_pct": target_pct,
            "stop_loss_pct": stop_loss_pct,
            "look_forward_days": look_forward_days,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with open(filepath, "wb") as f:
            pickle.dump(artifact, f)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

    msg = f"Model saved locally: {filepath}"
    _progress(msg)
    _progress_stats(
        "saved",
        msg,
        {"model_path": filepath},
    )
    
    # Upload to Supabase Storage (bucket 'ai-models') if enabled
    if upload_to_cloud:
        try:
            msg = "Uploading model to Supabase storage..."
            _progress(msg)
            _progress_stats("uploading", msg, {})
            with open(filepath, "rb") as f:
                supabase.storage.from_("ai-models").upload(
                    path=filename,
                    file=f,
                    file_options={"cache-control": "3600", "upsert": "true"}
                )
            msg = f"Model uploaded to Supabase Storage: ai-models/{filename}"
            _progress(msg)
            _progress_stats("uploaded", msg, {"model_path": filename})
        except Exception as e:
            print(f"Failed to upload model: {e}")
            # If bucket doesn't exist, this might fail. In reality, the user should create it.
            # But we can try to log it clearly.

if __name__ == "__main__":
    parser.add_argument("--exchange", required=True, help="Exchange name to train on")
    parser.add_argument("--max_features", type=int, help="Optional: maximum number of features to use for training")
    parser.add_argument("--grid", action="store_true", help="Enable GridSearchCV for hyperparameter optimization")
    args = parser.parse_args()
    
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("Error: NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required.")
        sys.exit(1)
        
    train_model(args.exchange, url, key, max_features_override=args.max_features, use_grid=args.grid)
