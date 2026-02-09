import argparse
import sys
import os
import pickle
import warnings
import re
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load envs similar to main.py
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))
# Also try web/.env.local where Supabase keys overlap often in this project structure
load_dotenv(os.path.join(base_dir, "web", ".env.local"), override=True)

if os.getenv("BT_DEBUG_ENV") == "1":
    print(
        f"DEBUG: NEXT_PUBLIC_SUPABASE_URL found? {'NEXT_PUBLIC_SUPABASE_URL' in os.environ}",
        flush=True,
    )


# Add api parent dir to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.stock_ai import _get_exchange_bulk_data, _get_exchange_bulk_intraday_data, _MetaLabelingClassifier
from api.train_exchange_model import add_massive_features

warnings.filterwarnings("ignore")

# Force UTF-8 stdout for Windows terminals to handle emojis
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Global cache for index data to avoid repeated file reads
_INDEX_CACHE = {}

_DATE_ISO_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}$")
_DATE_ISO_DATETIME_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}T")
_DATE_DMY_SLASH_RE = re.compile(r"^\\d{1,2}/\\d{1,2}/\\d{4}$")


def _parse_cli_date(value: str) -> pd.Timestamp:
    """
    Parse CLI dates reliably.

    Notes:
    - The UI sends ISO dates like YYYY-MM-DD. Pandas with dayfirst=True will mis-parse
      these as YYYY-DD-MM (e.g. 2025-07-01 -> 2025-01-07), so we must detect ISO.
    - We still support dd/mm/yyyy from older UI inputs.
    """
    v = (value or "").strip()
    if not v:
        raise ValueError("Empty date")

    if _DATE_ISO_RE.match(v):
        return pd.to_datetime(v, format="%Y-%m-%d", errors="raise")

    if _DATE_ISO_DATETIME_RE.match(v):
        return pd.to_datetime(v, errors="raise")

    if _DATE_DMY_SLASH_RE.match(v):
        return pd.to_datetime(v, dayfirst=True, errors="raise")

    # Fallback: prefer month-first parsing, then day-first.
    try:
        return pd.to_datetime(v, dayfirst=False, errors="raise")
    except Exception:
        return pd.to_datetime(v, dayfirst=True, errors="raise")


def load_egx30_index(start_date: str = None, end_date: str = None):
    """
    Load EGX30 index data from JSON file and filter by date range.
    Returns DataFrame with date index and close prices.
    """
    global _INDEX_CACHE
    
    # Check cache first
    cache_key = f"{start_date}:{end_date}"
    if cache_key in _INDEX_CACHE:
        return _INDEX_CACHE[cache_key].copy()
    
    try:
        # Path to EGX30-INDEX.json
        index_path = os.path.join(base_dir, "symbols_data", "EGX30-INDEX.json")
        
        if not os.path.exists(index_path):
            print(f"WARNING: EGX30 index file not found at {index_path}", flush=True)
            return None
        
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        df = pd.DataFrame(index_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # Cache the result
        _INDEX_CACHE[cache_key] = df.copy()
        
        print(f"DEBUG: Loaded EGX30 index data: {len(df)} days from {df.index[0]} to {df.index[-1]}", flush=True)
        return df
    
    except Exception as e:
        print(f"ERROR loading EGX30 index: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def calculate_benchmark_returns(start_date: str, end_date: str):
   
    index_df = load_egx30_index(start_date, end_date)

    if index_df is None or len(index_df) < 2:
        print("WARNING: Insufficient index data for benchmark calculation", flush=True)
        return None

    try:
        if 'open' in index_df.columns:
            start_price = float(index_df['open'].iloc[0])
        else:
            start_price = float(index_df['close'].iloc[0])
    except Exception:
        start_price = float(index_df['close'].iloc[0])

    end_price = float(index_df['close'].iloc[-1])

    benchmark_return = (end_price - start_price) / start_price
    
    print(
        f"DEBUG: EGX30 Benchmark - Start: {start_price:.2f}, End: {end_price:.2f}, "
        f"Return: {benchmark_return*100:.2f}%",
        flush=True
    )
    
    return benchmark_return

def load_model(model_path):
    """Loads a model from a pickle file."""
    try:
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        
        # Determine if it's a naked booster/model or a dictionary artifact
        if isinstance(obj, dict):
             # Check if it's a meta-labeling system
            if obj.get("kind") == "meta_labeling_system":
                return obj 
            return obj
        return obj
    except Exception as e:
        print(f"Error loading model {model_path}: {e}", flush=True)
        return None

def reconstruct_meta_model(artifact):
    """
    Reconstructs a usable _MetaLabelingClassifier from the dictionary artifact.
    """
    if not isinstance(artifact, dict) or artifact.get("kind") != "meta_labeling_system":
        return None
        
    import lightgbm as lgb
    
    pm_art = artifact["primary_model"]
    if pm_art.get("model_str"):
        primary_booster = lgb.Booster(model_str=pm_art["model_str"])
        # Wrap it if needed to have a similar API
        class PrimaryWrapper:
            def __init__(self, b): self.b = b
            def predict(self, X): return self.b.predict(X)
            def predict_proba(self, X): 
                raw = self.b.predict(X)
                return np.column_stack([1-raw, raw])
            
        primary_model = PrimaryWrapper(primary_booster)
    else:
        # Fallback
        primary_model = pm_art
        
    meta_model = artifact["meta_model"]
    meta_features = artifact["meta_feature_names"]
    threshold = artifact.get("meta_threshold", 0.6)
    
    from api.stock_ai import _MetaLabelingClassifier
    wrapper = _MetaLabelingClassifier(
        primary_model=primary_model,
        meta_model=meta_model,
        meta_feature_names=meta_features,
        meta_threshold=threshold
    )
    return wrapper

def run_radar_simulation(
    df,
    model,
    council=None,
    threshold=0.6,
    capital=100000,
    sim_start_dt: datetime | None = None,
    sim_end_dt: datetime | None = None,
    quiet: bool = False,
    validator_threshold: float | None = None,
    target_pct_override: float | None = None,
    stop_loss_pct_override: float | None = None,
):
    """
    Simulation of Radar: Base Model Detector -> Meta Model Confirmation.
    """
    if df.empty: return {}
    if not quiet:
        print(f"DEBUG: run_radar_simulation called for {len(df)} rows", flush=True)

    max_score = 0.0
    max_consensus = 0.0
    max_validator = 0.0
    
    balance = capital
    trade_log = []
    
    # Default settings (daily equities)
    TARGET_PCT = 0.10
    STOP_LOSS_PCT = 0.05
    HOLD_MAX_BARS = 20

    # Trailing Stop Rules (long-only)
    # - When unrealized profit reaches +5%: raise stop to entry (break-even)
    # - When unrealized profit reaches +8%: raise stop to +5% (lock profit)
    USE_TRAILING = True
    TRAIL_BE_PCT = 0.05
    TRAIL_LOCK_TRIGGER_PCT = 0.08
    TRAIL_LOCK_PCT = 0.05

    # Override from model artifact metadata when available so backtests match training assumptions.
    if isinstance(model, dict):
        def _meta_get(name: str, default=None):
            if not isinstance(model, dict):
                return default
            v = model.get(name)
            if v is not None:
                return v
            pm = model.get("primary_model") if isinstance(model.get("primary_model"), dict) else {}
            if isinstance(pm, dict):
                return pm.get(name, default)
            return default

        try:
            m_target = _meta_get("target_pct")
            m_sl = _meta_get("stop_loss_pct")
            m_hold = _meta_get("look_forward_days")

            if m_target is not None and float(m_target) > 0:
                TARGET_PCT = float(m_target)
            if m_sl is not None and float(m_sl) > 0:
                STOP_LOSS_PCT = float(m_sl)
            if m_hold is not None and int(m_hold) > 0:
                HOLD_MAX_BARS = int(m_hold)

            m_use_intraday = bool(_meta_get("use_intraday", False))
            m_tf = str(_meta_get("timeframe", "") or "").strip().lower()
            if m_use_intraday and m_tf in {"1m", "1h"}:
                USE_TRAILING = False
        except Exception:
            pass

    # UI/CLI Overrides (highest precedence)
    if target_pct_override is not None and target_pct_override > 0:
        TARGET_PCT = target_pct_override
    if stop_loss_pct_override is not None and stop_loss_pct_override > 0:
        STOP_LOSS_PCT = stop_loss_pct_override

    def _position_size_multiplier(score: float | None) -> float:
        """
        Map Validator Score -> position sizing multiplier.
        Score 0.40-0.55 => 0.5x
        Score 0.55-0.70 => 1.0x
        Score 0.70+     => 1.5x
        """
        try:
            if score is None or (isinstance(score, float) and np.isnan(score)):
                return 1.0
            s = float(score)
        except Exception:
            return 1.0

        if s < 0.55:
            return 0.5
        if s < 0.70:
            return 1.0
        return 1.5
    
    def _reset_booster_cats(obj):
        """Reset categorical features to avoid train/valid mismatch errors."""
        try:
            # Try to access the underlying LightGBM Booster in common wrappers.
            booster = (
                getattr(obj, "_Booster", None)
                or getattr(obj, "booster_", None)
                or getattr(obj, "booster", None)
                or getattr(obj, "b", None)  # PrimaryWrapper in this file
            )
            if booster is not None:
                # Set pandas_categorical to None to disable categorical feature checking
                if hasattr(booster, "pandas_categorical"):
                    booster.pandas_categorical = None
                # Also try to reset categorical_feature if it exists
                if hasattr(booster, "categorical_feature"):
                    booster.categorical_feature = "auto"
        except Exception as e:
            print(f"DEBUG: Could not reset booster cats: {e}", flush=True)

    def _reset_nested_boosters(obj):
        _reset_booster_cats(obj)
        for attr in ["primary_model", "meta_model", "model"]:
            child = getattr(obj, attr, None)
            if child is not None:
                _reset_booster_cats(child)

    def _get_primary_booster(obj):
        """Best-effort extraction of the underlying LightGBM Booster used for primary predictions."""
        try:
            pm = getattr(obj, "primary_model", None)
            if pm is None:
                return None
            return (
                getattr(pm, "_Booster", None)
                or getattr(pm, "booster_", None)
                or getattr(pm, "booster", None)
                or getattr(pm, "b", None)
            )
        except Exception:
            return None

    def _align_pandas_categories_to_booster(X_in: pd.DataFrame, cat_cols: list, booster, cat_cols_order: list):
        """
        If the booster has training-time pandas categories, coerce prediction categories to match.
        This avoids LightGBM's: "train and valid dataset categorical_feature do not match."
        """
        if X_in is None or X_in.empty or not cat_cols:
            return X_in

        if booster is None or not hasattr(booster, "pandas_categorical"):
            return X_in

        train_cats = getattr(booster, "pandas_categorical", None)
        if not isinstance(train_cats, list) or not train_cats:
            return X_in

        # We only know how to map categories positionally if we have the training categorical column order.
        if not cat_cols_order or len(train_cats) != len(cat_cols_order):
            return X_in

        mapping = {c: train_cats[i] for i, c in enumerate(cat_cols_order)}
        out = X_in.copy()
        for c in cat_cols:
            if c not in out.columns or c not in mapping:
                continue
            try:
                categories = [str(v) for v in list(mapping[c])]
                # Unknown categories become NaN (-1) which LightGBM treats as missing.
                out[c] = pd.Categorical(out[c].astype(str), categories=categories)
            except Exception:
                # If coercion fails, keep whatever we had.
                pass
        return out

    classifier = model
    if isinstance(model, dict) and model.get("kind") == "meta_labeling_system":
        classifier = reconstruct_meta_model(model)
        if not classifier:
            return {}
    
    def _align_for_king(X_src: pd.DataFrame, king_artifact: dict) -> pd.DataFrame:
        try:
            pm = king_artifact.get("primary_model") or {}
            feats = list(pm.get("feature_names") or [])
            cats = list(pm.get("categorical_features") or [])
            if not feats:
                return X_src.replace([np.inf, -np.inf], np.nan).fillna(0)
            Xk = X_src.copy()
            missing = [c for c in feats if c not in Xk.columns]
            for c in missing:
                Xk[c] = 0
            Xk = Xk[feats]
            for col in cats:
                if col in Xk.columns:
                    Xk[col] = Xk[col].astype(str).replace(['nan', 'None', ''], "Unknown").fillna("Unknown").astype('category')

            non_cat_cols = [c for c in Xk.columns if c not in set(cats)]
            for col in non_cat_cols:
                if not pd.api.types.is_numeric_dtype(Xk[col]):
                    Xk[col] = pd.to_numeric(Xk[col], errors="coerce")

            Xk = Xk.replace([np.inf, -np.inf], np.nan)
            if non_cat_cols:
                Xk[non_cat_cols] = Xk[non_cat_cols].fillna(0)
            return Xk
        except Exception:
            return X_src.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Pre-calculate signals
    try:
        expected_features = []
        categorical_features = []
        cat_cols = []

        if isinstance(model, dict) and model.get("kind") == "meta_labeling_system":
            pm = model.get("primary_model") or {}
            expected_features = list(pm.get("feature_names") or [])
            categorical_features = list(pm.get("categorical_features") or [])

        X_all = df.copy()
        X = X_all
        if expected_features:
            # Fill missing with 0 and subset
            missing = set(expected_features) - set(X.columns)
            for m in missing: X[m] = 0
            X = X[expected_features]
            
            # Decide categorical columns:
            # - Prefer the saved artifact list (pm.categorical_features).
            # - Fallback only when the feature is explicitly part of the model input.
            fallback_cats = [c for c in ["sector", "industry"] if c in expected_features]
            cat_cols = list(dict.fromkeys(list(categorical_features or []) + fallback_cats))

            # Normalize categoricals
            for col in cat_cols:
                if col in X.columns:
                    X[col] = (
                        X[col]
                        .astype(str)
                        .replace(["nan", "None", "", "0", "0.0"], "Unknown")
                        .fillna("Unknown")
                        .astype("category")
                    )

            # Force non-categoricals to numeric (prevents accidental "object" columns)
            non_cat_cols = [c for c in X.columns if c not in set(cat_cols)]
            for col in non_cat_cols:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = pd.to_numeric(X[col], errors="coerce")

            X = X.replace([np.inf, -np.inf], np.nan)
            if non_cat_cols:
                X[non_cat_cols] = X[non_cat_cols].fillna(0)

        X_pred = X.copy()

        # If we can, align category *levels* to the training booster to keep predictions meaningful.
        # Otherwise, reset booster categorical state per-call to avoid hard crashes.
        primary_booster = _get_primary_booster(classifier)
        X_pred = _align_pandas_categories_to_booster(
            X_pred,
            cat_cols=[c for c in cat_cols if c in X_pred.columns],
            booster=primary_booster,
            cat_cols_order=list(categorical_features or []),
        )

        # Guard: LightGBM categorical_feature mismatch (ensure booster doesn't carry stale pandas_categorical)
        if primary_booster is None or not (categorical_features and hasattr(primary_booster, "pandas_categorical")):
            _reset_nested_boosters(classifier)

        try:
            probs = classifier.predict_proba(X_pred)
            confidences = probs[:, 1]
        except Exception as e:
            print(f"[BT-LIVE] ERROR: run_radar_simulation prediction failed: {e}", flush=True)
            # Try converting all data to float to bypass categorical issues
            try:
                X_numeric = X_pred.copy()
                for col in X_numeric.columns:
                    if X_numeric[col].dtype == 'category' or X_numeric[col].dtype == 'object':
                        X_numeric[col] = X_numeric[col].astype('category').cat.codes.astype('float32')
                    else:
                        X_numeric[col] = X_numeric[col].astype('float32')
                
                _reset_nested_boosters(classifier)
                probs = classifier.predict_proba(X_numeric)
                confidences = probs[:, 1]
                print(f"[BT-LIVE] Recovered from prediction error by converting to numeric", flush=True)
            except Exception as e2:
                print(f"[BT-LIVE] ERROR: Failed to recover from prediction error: {e2}", flush=True)
                return {}
        
        # Phase 2: Council Filtering (use full feature frame so each model can align its own features)
        consensus_scores = None
        detailed_votes = {}
        if council:
            consensus_scores = council.get_consensus(X_all)
            detailed_votes = council.get_detailed_votes(X_all)

        # Optional: Council Validator (trains on KING BUYs; needs KING confidence feature)
        validator_probs = None
        if hasattr(run_radar_simulation, "_validator") and getattr(run_radar_simulation, "_validator", None) is not None:
            validator = getattr(run_radar_simulation, "_validator", None)
            # Get KING artifact/model from the council if present; otherwise allow a standalone KING for validator-only mode.
            king_obj = None
            try:
                king_obj = getattr(council, "models", {}).get("king") if council else None
            except Exception:
                king_obj = None
            if king_obj is None:
                king_obj = getattr(run_radar_simulation, "_king_validator_artifact", None)

            king_clf = king_obj
            if isinstance(king_clf, dict) and king_clf.get("kind") == "meta_labeling_system":
                king_clf = reconstruct_meta_model(king_clf)

            if king_clf is not None and hasattr(king_clf, "predict_proba"):
                # Align KING input to its training feature schema.
                Xk = _align_for_king(X_all, king_obj) if isinstance(king_obj, dict) else X_all

                # Critical: if KING was trained with pandas categoricals, LightGBM requires
                # the prediction-time category levels to match training-time levels.
                try:
                    if isinstance(king_obj, dict):
                        king_pm = king_obj.get("primary_model") or {}
                        king_cat_cols_order = list(king_pm.get("categorical_features") or [])
                        king_cat_cols = [c for c in king_cat_cols_order if c in Xk.columns]
                    else:
                        king_cat_cols_order = []
                        king_cat_cols = []

                    king_primary_booster = _get_primary_booster(king_clf)
                    Xk = _align_pandas_categories_to_booster(
                        Xk,
                        cat_cols=king_cat_cols,
                        booster=king_primary_booster,
                        cat_cols_order=king_cat_cols_order,
                    )
                except Exception:
                    pass

                king_conf = king_clf.predict_proba(Xk)[:, 1]
                try:
                    validator_probs = validator.predict_proba(X_all, primary_conf=king_conf)[:, 1]
                except Exception as e:
                    print(f"Warning: Council validator failed: {e}", flush=True)
                    validator_probs = None

        # Deep Debug (Disabled for clean terminal)
        # max_score = float(np.max(confidences)) if len(confidences) > 0 else 0.0
        # max_consensus = float(np.max(consensus_scores)) if consensus_scores is not None else 0.0
        # max_validator = float(np.max(validator_probs)) if validator_probs is not None and len(validator_probs) > 0 else 0.0
        
        # print(
        #     f"DEBUG: Symbol {df['symbol'].iloc[0] if 'symbol' in df.columns else '??'} "
        #     f"Max Radar: {max_score:.4f} | Max Council: {max_consensus:.4f} | Max Validator: {max_validator:.4f}",
        #     flush=True
        # )
        pass
        
    except Exception as e:
        print(f"ERROR: run_radar_simulation prediction failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {}

    # Iterate to trade
    dates = df.index

    # Accept either lowercase (close/high/low) or the more common OHLCV casing (Close/High/Low)
    close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
    high_col = "high" if "high" in df.columns else ("High" if "High" in df.columns else None)
    low_col = "low" if "low" in df.columns else ("Low" if "Low" in df.columns else None)
    if close_col is None or high_col is None or low_col is None:
        missing_ohlc = [c for c, v in [("close/Close", close_col), ("high/High", high_col), ("low/Low", low_col)] if v is None]
        print(f"ERROR: Missing OHLC columns for trading loop: {missing_ohlc}", flush=True)
        return {}

    closes = df[close_col].values
    highs = df[high_col].values
    lows = df[low_col].values
    symbols = df['symbol'].values if 'symbol' in df.columns else [None]*len(df)
    
    # Track pre-council and post-council separately
    pre_council_trades = []
    post_council_trades = []
    balance_pre = capital
    balance_post = capital
    
    hold_max = min(HOLD_MAX_BARS, max(1, len(df) - 1))
    if hold_max != HOLD_MAX_BARS:
        print(f"[BT-LIVE] Using adaptive hold window: {hold_max} bars (requested {HOLD_MAX_BARS})", flush=True)

    in_trade = False
    exit_idx = -1

    for i in range(len(df) - hold_max):
        # Skip if we are currently in a trade
        if in_trade:
            if i <= exit_idx:
                continue
            else:
                in_trade = False
        radar_score = confidences[i]
        council_score = consensus_scores[i] if consensus_scores is not None else 1.0
        validator_score = float(validator_probs[i]) if validator_probs is not None else None
        
        # Check if Radar phase passes (before council)
        passes_radar = radar_score >= threshold
        
        # Check if Council phase also passes
        passes_council = council_score >= 0.55
        if validator_probs is not None:
            v_thresh = validator_threshold
            if v_thresh is None:
                v_thresh = float(getattr(getattr(run_radar_simulation, "_validator", None), "approval_threshold", 0.5))
            
            passes_council = passes_council and (validator_probs[i] >= v_thresh)
        
        # Track pre-council if radar passes
        if passes_radar:
            score = council_score # Use consensus as the final "score" for the log
            entry_price = closes[i]
            entry_date = dates[i]
            symbol = symbols[i]

            try:
                entry_dt = pd.to_datetime(entry_date).tz_localize(None)
            except Exception:
                entry_dt = None

            if sim_start_dt is not None and entry_dt is not None and entry_dt < sim_start_dt:
                continue
            if sim_end_dt is not None and entry_dt is not None and entry_dt > sim_end_dt:
                continue
            
            take_profit = entry_price * (1 + TARGET_PCT)
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)
            current_stop = float(stop_loss)
            trail_mode = "NONE"
            
            outcome = "HOLD"
            pnl_pct = 0.0
            exit_date = dates[i+hold_max]
            exit_price = closes[i+hold_max]
            exit_idx = i + hold_max
            
            for days_fwd in range(1, hold_max + 1):
                idx = i + days_fwd
                if idx >= len(df): break

                hi = float(highs[idx])
                lo = float(lows[idx])
                
                # Conservative bar evaluation:
                # - Use the stop that was active coming into this bar.
                # - If target hits, we exit at target.
                # - Trailing-stop updates are applied AFTER this bar (effective next bar).
                if lo <= current_stop:
                    outcome = "STOP LOSS ❌" if trail_mode == "NONE" else f"TRAIL STOP ({trail_mode}) 🛡️"
                    pnl_pct = (current_stop - entry_price) / entry_price
                    exit_date = dates[idx]
                    exit_price = current_stop
                    exit_idx = idx
                    break

                if hi >= take_profit:
                    outcome = "TARGET HIT 🎯"
                    pnl_pct = TARGET_PCT
                    exit_date = dates[idx]
                    exit_price = take_profit
                    exit_idx = idx
                    break

                if USE_TRAILING:
                    # Update trailing stop based on achieved profit (effective next bar)
                    be_price = float(entry_price)
                    lock_price = float(entry_price * (1 + TRAIL_LOCK_PCT))
                    if hi >= float(entry_price * (1 + TRAIL_LOCK_TRIGGER_PCT)) and current_stop < lock_price:
                        current_stop = lock_price
                        trail_mode = "+5%"
                    elif hi >= float(entry_price * (1 + TRAIL_BE_PCT)) and current_stop < be_price:
                        current_stop = be_price
                        trail_mode = "BE"
            
            if outcome == "HOLD":
                pnl_pct = (exit_price - entry_price) / entry_price
                outcome = f"TIME EXIT ({pnl_pct*100:.1f}%)"

            try:
                days_held = int((pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days)
            except Exception:
                days_held = 0

            sizing_score = validator_score if validator_score is not None else float(score)
            size_mult = _position_size_multiplier(sizing_score)
            try:
                fs = None
                if "fund_score" in df.columns:
                    fs = df["fund_score"].iloc[i]
                if fs is not None and (isinstance(fs, float) and np.isnan(fs)):
                    fs = None
            except Exception:
                fs = None
            trade_data = {
                "Date": entry_date.strftime("%d/%m/%Y") if hasattr(entry_date, "strftime") else str(entry_date),
                "Entry_Date": entry_date.strftime("%Y-%m-%d"),
                "Exit_Date": exit_date.strftime("%Y-%m-%d"),
                "Entry_Day": entry_date.strftime("%A"),
                "Exit_Day": exit_date.strftime("%A"),
                "Days_Held": days_held,
                "Symbol": symbol,
                "Entry": entry_price,
                "Exit": exit_price,
                "Score": round(float(score), 2),
                "Radar_Score": round(float(radar_score), 4),
                "Validator_Score": (round(float(validator_score), 4) if validator_score is not None else None),
                "Sizing_Score": round(float(sizing_score), 4),
                "Size_Multiplier": float(size_mult),
                "Fund_Score": (float(fs) if fs is not None else None),
                "Result": outcome,
                "PnL_Pct": float(pnl_pct),
                "Status": "Accepted" if passes_council else "Rejected",
                "Votes": {m: round(float(v[i]), 2) for m, v in detailed_votes.items()}
            }
            
            # Set in_trade flag
            in_trade = True

            # Track pre-council (all radar signals)
            balance_pre *= (1 + pnl_pct)
            pre_council_trades.append({**trade_data, "Balance": int(balance_pre)})
            
            # Only track post-council if passes both filters
            if passes_council:
                balance_post *= (1 + pnl_pct)
                post_council_trades.append({**trade_data, "Balance": int(balance_post)})
                trade_log.append({**trade_data, "Balance": int(balance_post)})
            else:
                # Still add it to a "global log" if we want to show rejected trades in dialog
                # Let's decide if trade_log should contain ALL trades with a status field
                # High complexity UI: yes, all trades.
                trade_log.append({**trade_data, "Balance": int(balance_post)})

    # Calculate metrics for both phases
    def calc_metrics(trades_list, ignore_status=False):
        if not trades_list:
            return {"total": 0, "win_rate": 0, "profit_pct": 0, "rejected_profitable": 0}
        
        # If ignore_status is True, we treat all trades as Valid candidates (Pre-Council view)
        # If False, we respect the 'Status' field (Post-Council view)
        
        if ignore_status:
            relevant_trades = trades_list
        else:
            relevant_trades = [t for t in trades_list if t.get("Status") == "Accepted"]

        # 1. Correct Win Rate Calculation: (Count Wins / Total Count) * 100
        # Check if 'PnL_Pct' > 0 for a win
        wins_count = sum(1 for t in relevant_trades if t["PnL_Pct"] > 0)
        total_count = len(relevant_trades)
        
        win_rate = (wins_count / total_count * 100) if total_count > 0 else 0.0
        
        # 2. Correct Rejected Profitable Calculation
        # Count trades that were Rejected but turned out to have Positive PnL
        rejected_profitable_count = sum(1 for t in trades_list if t.get("Status") == "Rejected" and t["PnL_Pct"] > 0)
        
        # 3. Profit Calculation
        # We use the final balance from the simulation loop
        final_bal = trades_list[-1]["Balance"] if trades_list else capital
        total_return = (final_bal - capital) / capital * 100
        
        return {
            "total": total_count, 
            "win_rate": win_rate, 
            "profit_pct": total_return, 
            "rejected_profitable": rejected_profitable_count
        }
    
    pre_metrics = calc_metrics(pre_council_trades, ignore_status=True)
    post_metrics = calc_metrics(post_council_trades, ignore_status=False)
    
    # We actually want total "Rejected Profitable" globally
    # This is now calculated within calc_metrics for each list and aggregated in main()
    # rejected_profitable = sum(1 for t in trade_log if t.get("Status") == "Rejected" and t["PnL_Pct"] > 0)

    return {
        "Total Trades": len([t for t in trade_log if t.get("Status") == "Accepted"]),
        "Trades Log": pd.DataFrame(trade_log),
        "pre_council_trades": pre_metrics["total"],
        "pre_council_win_rate": pre_metrics["win_rate"],
        "pre_council_profit_pct": pre_metrics["profit_pct"],
        "post_council_trades": post_metrics["total"],
        "post_council_win_rate": post_metrics["win_rate"],
        "post_council_profit_pct": post_metrics["profit_pct"],
        "rejected_profitable": pre_metrics["rejected_profitable"], # Use the one from pre_metrics as it considers all radar signals
        "max_radar": max_score,
        "max_council": max_consensus,
        "max_validator": max_validator,
        "threshold_used": float(threshold),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default=None, help="Optional CSV output path")
    parser.add_argument("--council", default=None, help="Path to council model (enables Council filtering)")
    parser.add_argument("--validator", default=None, help="Optional Council Validator model (trained on KING BUYs)")
    parser.add_argument("--meta-threshold", type=float, default=None, help="Override meta threshold (0-1)")
    parser.add_argument("--validator-threshold", type=float, default=None, help="Override validator threshold (0-1)")
    parser.add_argument("--target-pct", type=float, default=None, help="Override target profit percentage (e.g. 0.15)")
    parser.add_argument("--stop-loss-pct", type=float, default=None, help="Override stop loss percentage (e.g. 0.05)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose debug output")
    parser.add_argument("--no-trades-json", action="store_true", help="Do not print trades JSON to stdout")
    args = parser.parse_args()

    # Always include a buffer window before the selected start date
    # so indicators have enough history (trades still limited to start/end).
    SIM_BUFFER_DAYS = 90
    
    # Load Data
    try:
        start_dt = _parse_cli_date(args.start).tz_localize(None)
        args.start = start_dt.strftime("%Y-%m-%d") 
        
        # Use a generous buffer (1 year) to ensure all technical indicators have enough history
        buffer_start_dt = start_dt - timedelta(days=365)
        buffer_start = buffer_start_dt.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Warning: Date parsing failed ({e}), using defaults.", flush=True)
        buffer_start = "2023-01-01"
        start_dt = pd.to_datetime("2024-01-01")

    # Load Model (needed early so we can match data source/timeframe)
    models_dir = os.path.join(base_dir, "api", "models")
    model_path = os.path.join(models_dir, args.model)
    if not os.path.exists(model_path):
        if os.path.exists(args.model):
            model_path = args.model
        else:
            print(f"❌ Model not found: {model_path}", flush=True)
            return

    print(f"🧠 Loading model: {args.model}...", flush=True)
    model_obj = load_model(model_path)
    if not model_obj:
        return

    def _meta_get(obj, name: str, default=None):
        if not isinstance(obj, dict):
            return default
        v = obj.get(name)
        if v is not None:
            return v
        pm = obj.get("primary_model") if isinstance(obj.get("primary_model"), dict) else {}
        if isinstance(pm, dict):
            return pm.get(name, default)
        return default

    # Meta threshold (UI override > model > default)
    sim_threshold = 0.40
    if args.meta_threshold is not None:
        try:
            sim_threshold = float(args.meta_threshold)
            print(f"🎯 Using Meta Threshold: {sim_threshold} (UI override)", flush=True)
        except Exception:
            sim_threshold = 0.40
            print(f"⚠️ Invalid meta-threshold provided. Using default {sim_threshold}.", flush=True)
    elif isinstance(model_obj, dict):
        sim_threshold = float(_meta_get(model_obj, "meta_threshold", sim_threshold))
        print(f"🎯 Using Meta Threshold: {sim_threshold}", flush=True)
    else:
        print(f"🎯 Using Meta Threshold: {sim_threshold}", flush=True)

    # Decide data source based on model metadata (fallback: CRYPTO => 1h intraday)
    use_intraday = False
    timeframe = "1d"
    try:
        use_intraday = bool(_meta_get(model_obj, "use_intraday", False))
        timeframe = str(_meta_get(model_obj, "timeframe", "1d") or "1d").strip().lower()
    except Exception:
        use_intraday = False
        timeframe = "1d"

    if args.exchange.strip().upper() == "CRYPTO" and not use_intraday:
        use_intraday = True
        timeframe = "1h"

    if use_intraday:
        print(
            f"📥 Fetching intraday bulk data for {args.exchange} ({timeframe}) (Buffer Start: {buffer_start}, Sim Start: {args.start})...",
            flush=True,
        )
        data_map = _get_exchange_bulk_intraday_data(args.exchange, timeframe=timeframe, from_ts=buffer_start)
    else:
        print(
            f"📥 Fetching bulk data for {args.exchange} (Buffer Start: {buffer_start}, Sim Start: {args.start})...",
            flush=True,
        )
        data_map = _get_exchange_bulk_data(args.exchange, from_date=buffer_start)

    if not data_map:
        print("❌ No data found.", flush=True)
        return

    # Context & Fundamentals
    from api.train_exchange_model import add_market_context, fetch_fundamentals_for_exchange
    from api.stock_ai import _init_supabase, supabase

    _init_supabase()

    market_df = None
    if args.exchange == "EGX":
        try:
            index_path = os.path.join(base_dir, "symbols_data", "EGX30-INDEX.json")
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    idx_data = json.load(f)
                market_df = pd.DataFrame(idx_data)
                market_df['date'] = pd.to_datetime(market_df['date'])
                market_df.set_index('date', inplace=True)
                print("✅ Market context (EGX30) loaded from local JSON.", flush=True)
        except Exception:
            pass

    df_funds = pd.DataFrame()
    if supabase and args.exchange != "CRYPTO":
        print(f"📥 Fetching fundamentals for {args.exchange}...", flush=True)
        df_funds = fetch_fundamentals_for_exchange(supabase, args.exchange)

    # Prepare TheCouncil (Phase 2) ONLY when explicitly requested.
    # If the user chooses "None (No Filter)" in the UI, the API won't pass --council,
    # and we must not fall back to any default Council model.
    # Clear optional attachments (each run should be independent)
    try:
        setattr(run_radar_simulation, "_validator", None)
        setattr(run_radar_simulation, "_king_validator_artifact", None)
    except Exception:
        pass

    council = None
    council_arg = (args.council or "").strip()
    if council_arg and council_arg.lower() not in {"none", "null", "no", "no_filter", "no filter"}:
        from api.council import TheCouncil

        council_models = {"collector": model_obj}

        actual_council_path = os.path.join(models_dir, council_arg)
        if os.path.exists(actual_council_path):
            print(f"🏛️ Loading Council Model: {council_arg}...", flush=True)
            loaded = load_model(actual_council_path)
            # Guard: user may accidentally pick a Council Validator model in the council dropdown.
            if isinstance(loaded, dict) and (loaded.get("kind") or "").strip().lower() == "council_validator":
                print("⚠️ Selected model is a Council Validator, not a Council member. Using it as --validator instead.", flush=True)
                if not args.validator:
                    args.validator = council_arg
                loaded = None
            council_models["king"] = loaded
        elif os.path.exists(council_arg):
            print(f"🏛️ Loading Council Model (abs): {council_arg}...", flush=True)
            loaded = load_model(council_arg)
            if isinstance(loaded, dict) and (loaded.get("kind") or "").strip().lower() == "council_validator":
                print("⚠️ Selected model is a Council Validator, not a Council member. Using it as --validator instead.", flush=True)
                if not args.validator:
                    args.validator = council_arg
                loaded = None
            council_models["king"] = loaded
        else:
            print(f"⚠️ Council model not found at {actual_council_path}. Council will be disabled.", flush=True)

        if council_models.get("king") is not None:
            council = TheCouncil(models_dict=council_models)
        else:
            council = None
    else:
        print("🏛️ Council disabled (No Filter).", flush=True)

    # Optional validator (gates Council-approved trades based on KING confidence)
    if args.validator:
        from api.council_validator import load_council_validator_from_path
        v_path = os.path.join(models_dir, args.validator)
        if not os.path.exists(v_path) and os.path.exists(args.validator):
            v_path = args.validator
        validator = load_council_validator_from_path(v_path)
        if validator:
            # Attach to function for minimal signature changes
            setattr(run_radar_simulation, "_validator", validator)
            print(f"🛡️ Loaded Council Validator: {os.path.basename(v_path)}", flush=True)

            # If Council is disabled, still support validator-only filtering by loading KING for confidence.
            if council is None:
                king_path = os.path.join(models_dir, "KING 👑.pkl")
                if os.path.exists(king_path):
                    king_art = load_model(king_path)
                    setattr(run_radar_simulation, "_king_validator_artifact", king_art)
                    print("👑 Loaded KING for validator-only mode.", flush=True)
        else:
            print(f"⚠️ Failed to load Council Validator from {v_path}", flush=True)

    # Running Simulation
    from api.train_exchange_model import add_technical_indicators, add_indicator_signals
    
    all_trades = []
    all_res_metadata = []
    count = 0
    symbols_list = list(data_map.keys())
    
    print(f"🚀 Processing {len(symbols_list)} symbols sequentially...", flush=True)
    
    for symbol in symbols_list:
        df = data_map[symbol]
        if df.empty:
            continue
        
        # Save original index
        original_index = df.index
        if not isinstance(original_index, pd.DatetimeIndex):
            original_index = pd.to_datetime(original_index)
        
        if len(df) < 60:
            continue
        
        try:
            df_feat = add_technical_indicators(df)
            if df_feat.empty:
                continue
            
            # Ensure index matches
            if len(df_feat) == len(df):
                df_feat.index = original_index
            
            df_feat = add_indicator_signals(df_feat)
            df_feat = add_massive_features(df_feat)
            
            if market_df is not None:
                df_feat = add_market_context(df_feat, market_df)
            
            df_feat['symbol'] = symbol
            if not df_funds.empty:
                df_feat = df_feat.join(df_funds.set_index("symbol"), on="symbol", how="left")
            
            if len(df_feat) == len(df):
                df_feat.index = original_index

            fund_score_raw = df_feat["fund_score"] if "fund_score" in df_feat.columns else None
            df_feat = df_feat.fillna(0)
            if fund_score_raw is not None:
                df_feat["fund_score"] = fund_score_raw
            
            # Slice simulation period (with optional buffer if too few rows)
            sim_start_dt = _parse_cli_date(args.start).tz_localize(None)
            sim_end_dt = _parse_cli_date(args.end).tz_localize(None) if args.end else None
            
            if not isinstance(df_feat.index, pd.DatetimeIndex):
                df_feat.index = pd.to_datetime(df_feat.index, errors="coerce")
            idx_clean = pd.DatetimeIndex(df_feat.index).tz_localize(None)

            fmt = "%d/%m/%Y"
            
            mask = (idx_clean >= sim_start_dt)
            if sim_end_dt:
                mask = mask & (idx_clean <= sim_end_dt)
            
            buffer_start_dt = sim_start_dt - timedelta(days=SIM_BUFFER_DAYS)
            buffer_mask = (idx_clean >= buffer_start_dt)
            if sim_end_dt:
                buffer_mask = buffer_mask & (idx_clean <= sim_end_dt)
            df_sim = df_feat[buffer_mask]
            # print(
            #     f"[BT-LIVE] Buffer applied for {symbol}: {len(df_sim)} rows "
            #     f"(buffer {SIM_BUFFER_DAYS}d) — trades limited to {args.start} → {args.end}",
            #     flush=True
            # )
            
            if df_sim.empty:
                continue

            res = run_radar_simulation(
                df_sim,
                model_obj,
                council=council,
                threshold=sim_threshold,
                sim_start_dt=sim_start_dt,
                sim_end_dt=sim_end_dt,
                quiet=args.quiet,
                validator_threshold=args.validator_threshold,
                target_pct_override=args.target_pct,
                stop_loss_pct_override=args.stop_loss_pct,
            ) 
            
            if isinstance(res, dict) and res:
                # Always keep metadata (even when no trades) for diagnostics / aggregate stats.
                all_res_metadata.append(res)

                if res.get("Trades Log") is not None and not res["Trades Log"].empty:
                    all_trades.append(res["Trades Log"])
                
        except Exception as e:
            print(f"CRITICAL Error processing {symbol}: {e}", flush=True)
            
        count += 1
        if count % 20 == 0:
            print(f"Progress: {count}/{len(symbols_list)} symbols processed...", flush=True)

    # Global Report
    if not all_trades:
        max_radar = max((float(r.get("max_radar") or 0.0) for r in all_res_metadata), default=0.0)
        threshold_used = None
        try:
            threshold_used = float(all_res_metadata[0].get("threshold_used")) if all_res_metadata else None
        except Exception:
            threshold_used = None

        if threshold_used is not None:
            print(
                f"❌ No trades found matching criteria. (Processed {len(symbols_list)} symbols) "
                f"| Max Radar={max_radar:.4f} | Threshold={threshold_used:.4f}",
                flush=True,
            )
        else:
            print(
                f"❌ No trades found matching criteria. (Processed {len(symbols_list)} symbols) "
                f"| Max Radar={max_radar:.4f}",
                flush=True,
            )

        # Still emit the JSON marker block so the API/UI can show an empty log consistently.
        if not args.no_trades_json:
            print("\n--- JSON TRADES LOG START ---", flush=True)
            print("[]", flush=True)
            print("--- JSON TRADES LOG END ---", flush=True)
        return
        
    global_log = pd.concat(all_trades).sort_values("Date")
    capital_per_trade = 10000

    # Dynamic Position Sizing (based on Validator Score when available, otherwise Council Score)
    # Profit_Cash is computed using a fixed base notional per trade (capital_per_trade) times a size multiplier.
    accepted_mask = global_log.get("Status", "Accepted").fillna("Accepted").astype(str).str.lower().eq("accepted")
    if "Size_Multiplier" in global_log.columns:
        base_notional = capital_per_trade * global_log["Size_Multiplier"].fillna(1.0)
    else:
        base_notional = float(capital_per_trade)

    # Rejected trades are not executed => no P/L contribution.
    global_log["Position_Cash"] = np.where(accepted_mask, base_notional, 0.0)
    global_log['Profit_Cash'] = global_log['Position_Cash'] * global_log['PnL_Pct']
    global_log['Cumulative_Profit'] = global_log['Profit_Cash'].cumsum()
    net_profit = global_log['Profit_Cash'].sum()
    denom = int(accepted_mask.sum()) if int(accepted_mask.sum()) > 0 else 1
    win_rate = (len(global_log[accepted_mask & (global_log['PnL_Pct'] > 0)]) / denom) * 100

    # Aggregate Council Impact
    total_pre_trades = sum(r.get("pre_council_trades", 0) for r in all_res_metadata)
    total_post_trades = sum(r.get("post_council_trades", 0) for r in all_res_metadata)
    
    # Calculate weighted averages or just re-calculate from logs if possible?
    # Simple aggregation of pre_metrics
    avg_pre_win_rate = (sum(r.get("pre_council_win_rate", 0) * r.get("pre_council_trades", 0) for r in all_res_metadata) / total_pre_trades) if total_pre_trades > 0 else 0
    avg_post_win_rate = (sum(r.get("post_council_win_rate", 0) * r.get("post_council_trades", 0) for r in all_res_metadata) / total_post_trades) if total_post_trades > 0 else 0
    
    # For profit %, it's cumulative. Let's just output the totals.
    total_pre_profit_pct = sum(r.get("pre_council_profit_pct", 0) for r in all_res_metadata) / len(all_res_metadata) # simplified
    total_post_profit_pct = sum(r.get("post_council_profit_pct", 0) for r in all_res_metadata) / len(all_res_metadata) # simplified

    # Aggregate rejected profitable
    rejected_profitable = sum(r.get("rejected_profitable", 0) for r in all_res_metadata)

    # Output JSON for API consumption
    if not args.no_trades_json:
        print("\n--- JSON TRADES LOG START ---", flush=True)
        print(global_log.to_json(orient="records", date_format="iso"), flush=True)
        print("--- JSON TRADES LOG END ---", flush=True)

    # out_file = args.out or f"backtest_results_{args.exchange}.csv"
    # try:
    #     global_log.to_csv(out_file, index=False, encoding="utf-8")
    # except Exception as e:
    #     print(f"Warning: Failed to write CSV {out_file}: {e}", flush=True)
    
    print("\n" + "="*40, flush=True)
    print(" 🚀 FINAL RADAR BACKTEST REPORT ", flush=True)
    print("="*40, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Exchange: {args.exchange}", flush=True)
    
    try:
        s_fmt = _parse_cli_date(args.start).strftime("%d/%m/%Y")
        e_fmt = _parse_cli_date(args.end).strftime("%d/%m/%Y") if args.end else "Present"
    except:
        s_fmt, e_fmt = args.start, (args.end or "Present")
        
    print(f"Period: {s_fmt} to {e_fmt}", flush=True)
    print("-" * 20, flush=True)
    print(f"Total Trades Detected: {total_post_trades}", flush=True)
    print(f"Win Rate:              {win_rate:.1f}%", flush=True)
    print(f"Avg Return per Trade:  {global_log['PnL_Pct'].mean()*100:.2f}%", flush=True)
    print("-" * 20, flush=True)
    print(f"Simulated Profit (Base 10k + Dynamic Sizing): {int(net_profit):,} EGP", flush=True)
    
    print("\n--- Council Impact Analysis ---", flush=True)
    print(f"Pre-Council Trades:    {total_pre_trades}", flush=True)
    print(f"Post-Council Trades:   {total_post_trades}", flush=True)
    print(f"Trades Filtered:       {total_pre_trades - total_post_trades} ({((total_pre_trades - total_post_trades)/total_pre_trades)*100:.1f}% reduction)" if total_pre_trades > 0 else "N/A", flush=True)
    print(f"Pre-Council Win Rate:  {avg_pre_win_rate:.1f}%", flush=True)
    print(f"Post-Council Win Rate: {avg_post_win_rate:.1f}%", flush=True)
    print(f"Pre-Council Profit:    {total_pre_profit_pct:.2f}%", flush=True)
    print(f"Post-Council Profit:   {total_post_profit_pct:.2f}%", flush=True)
    print(f"Win Rate Boost:        {avg_post_win_rate - avg_pre_win_rate:+.1f} percentage points", flush=True)
    print(f"Rejected Profitable:   {rejected_profitable}", flush=True)
    print("="*40, flush=True)


if __name__ == "__main__":
    main()
