import os
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, make_scorer
from supabase import create_client, Client

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
    
    # ---------------------------------------------------------
    # 2. Generate Rolling Windows - The magic multiplier
    # ---------------------------------------------------------
    # Calculate mean and standard deviation for different periods
    # (3 days for speculation, 7 weekly, 14, 30 for monthly)
    windows = [3, 7, 14, 30]
    target_cols = ['close', 'volume', 'momentum_rsi'] # Important columns for analysis
    extra_cols = {}
    
    for w in windows:
        for col in target_cols:
            if col in df.columns:
                extra_cols[f'{col}_SMA_{w}'] = df[col].rolling(window=w).mean()
                extra_cols[f'{col}_STD_{w}'] = df[col].rolling(window=w).std()
                extra_cols[f'{col}_MAX_{w}'] = df[col].rolling(window=w).max()
                extra_cols[f'{col}_MIN_{w}'] = df[col].rolling(window=w).min()

    # ---------------------------------------------------------
    # 3. Historical Memory (Lag Features)
    # ---------------------------------------------------------
    # What happened yesterday and the day before yesterday? (Very important for AI)
    lags = [1, 2, 3, 5]
    for lag in lags:
        extra_cols[f'Close_Lag_{lag}'] = df['close'].shift(lag)
        extra_cols[f'Vol_Lag_{lag}'] = df['volume'].shift(lag)
        extra_cols[f'Return_{lag}d'] = df['close'].pct_change(lag)

    # ---------------------------------------------------------
    # 4. Custom Advanced Indicators (Custom Math)
    # ---------------------------------------------------------
    # Log Returns (Better for statistical models)
    extra_cols['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Z-Score (Normalize price for the model to understand expensive and cheap stocks in the same way)
    extra_cols['Z_Score_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    
    # Price x Volume interaction
    extra_cols['PV_Trend'] = df['close'].pct_change() * df['volume'].pct_change()

    extra_cols["Close"] = df["close"]
    extra_cols["Volume"] = df["volume"]
    if "open" in df.columns:
        extra_cols["Open"] = df["open"]
    if "high" in df.columns:
        extra_cols["High"] = df["high"]
    if "low" in df.columns:
        extra_cols["Low"] = df["low"]

    if extra_cols:
        df = pd.concat([df, pd.DataFrame(extra_cols, index=df.index)], axis=1)
        df = df.copy()

    # Clean data (remove NaN values resulting from Lags)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
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

    return out

def prepare_for_ai(df):
    if df.empty: return df
    out = df.copy()
    out["Next_Close"] = out["Close"].shift(-1)
    out["Target"] = (out["Next_Close"] > out["Close"]).astype(int)
    out = out.dropna().copy()
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
):
    print(f"Starting training for exchange: {exchange}")
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
    except Exception:
        rows_total = None
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        res = (
            supabase.table("stock_prices")
            .select("*")
            .eq("exchange", exchange)
            .order("date", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = res.data or []
        if not batch:
            break
        all_rows.extend(batch)
        msg = f"Loaded {len(all_rows):,} rows so far"
        _progress_stats(
            "loading_rows",
            msg,
            {"rows_loaded": len(all_rows), "rows_total": rows_total},
        )
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

    all_symbols = sorted(df_all["symbol"].dropna().unique().tolist())
    raw_rows = len(df_all)
    msg = f"Loaded {raw_rows:,} rows for {len(all_symbols):,} symbols ({exchange})"
    _progress(msg)
    _progress_stats(
        "data_loaded",
        msg,
        {"raw_rows": raw_rows, "symbols_total": len(all_symbols), "rows_total": raw_rows},
    )

    combined_data = []
    symbols_used = 0
    symbols_total = len(all_symbols)
    symbols_processed = 0
    progress_step = max(1, symbols_total // 50) if symbols_total else 1
    for symbol, df_symbol in df_all.groupby("symbol"):
        symbols_processed += 1
        if df_symbol is None or df_symbol.empty:
            continue
        if len(df_symbol) < 200:
            continue

        df = df_symbol.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        if symbols_processed % progress_step == 0 or symbols_processed == symbols_total:
            msg = f"Processing symbol {symbol} ({symbols_processed}/{symbols_total})"
            _progress(msg)
            _progress_stats(
                "processing_symbols",
                msg,
                {"symbols_processed": symbols_processed, "symbols_total": symbols_total},
            )

        feat = add_massive_features(df)
        tech = add_technical_indicators(df)
        if not tech.empty:
            feat = feat.drop(columns=[c for c in tech.columns if c in feat.columns], errors="ignore")
            feat = pd.concat([feat, tech], axis=1)
        ready = prepare_for_ai(feat)

        if len(ready) >= 120:
            combined_data.append(ready)
            symbols_used += 1

    if not combined_data:
        _progress("No valid data collected for training")
        return

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

    # Use a large number of estimators with early stopping to prevent overfitting.
    if use_early_stopping:
        final_n_estimators = n_estimators or 1000
        msg = f"Training LightGBM (early_stopping=True, n_estimators={final_n_estimators})"
        _progress(msg)
        _progress_stats(
            "training",
            msg,
            {"n_estimators": final_n_estimators, "early_stopping": True},
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False,
        )
        # Debug: log final n_estimators for early-stopping mode
        print("Final n_estimators:", final_n_estimators)
        model = LGBMClassifier(
            n_estimators=final_n_estimators,
            random_state=1,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="logloss",
            callbacks=[
                # LightGBM early stopping via callback API
                lgb.early_stopping(stopping_rounds=50),
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
    
    # Persist a lightweight summary for the UI
    try:
        num_features = int(getattr(model, "n_features_in_", X.shape[1]))
        if num_features <= 20:
            print(f"Warning: model for {exchange} has only {num_features} features (expected > 20)")
        summary = {
            "exchange": exchange,
            "useEarlyStopping": use_early_stopping,
            "nEstimators": int(final_n_estimators),
            "trainingSamples": int(total_samples),
            "numFeatures": num_features,
            "featurePreset": preset,
            "symbolsUsed": int(symbols_used),
            "rawRows": int(raw_rows),
            "timestamp": datetime.utcnow().isoformat() + "Z",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", required=True, help="Exchange name to train on")
    parser.add_argument("--max_features", type=int, help="Optional: maximum number of features to use for training")
    args = parser.parse_args()
    
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("Error: NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required.")
        sys.exit(1)
        
    train_model(args.exchange, url, key, max_features_override=args.max_features)
