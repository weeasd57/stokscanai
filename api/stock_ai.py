import datetime as dt
import os
import urllib.request
import urllib.error
import pickle
import time
import warnings
import hashlib

# Suppress specific FutureWarnings from libraries like 'ta'
warnings.filterwarnings("ignore", category=FutureWarning)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from threading import Lock
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import json
from eodhd import APIClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from supabase import create_client, Client
from api.train_exchange_model import add_massive_features, add_market_context

# Conditional import for LGBM to avoid failure if not installed (though it should be)
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None


class _LgbmBoosterClassifier:
    def __init__(self, booster, threshold: float = 0.40): # Lowered to 0.4
        self.booster = booster
        try:
            self.threshold = float(threshold)
        except Exception:
            self.threshold = 0.40

    def predict(self, X):
        raw = self.booster.predict(X)
        return (np.asarray(raw) >= self.threshold).astype(int)

    def predict_proba(self, X):
        raw = self.booster.predict(X)
        probs = np.asarray(raw)
        # Return 2-column format: [prob_class_0, prob_class_1]
        return np.column_stack([1 - probs, probs])

    def get_feature_importance(self):
        return self.booster.feature_importance()

supabase: Optional[Client] = None

# Predictors used by the pre-trained LightGBM models
# Note: the actual predictors used at inference time are the intersection of
# this list and the columns present in the feature dataframe.
LGBM_PREDICTORS = [
    "Close", "Volume",
    "SMA_50", "SMA_200",
    "EMA_50", "EMA_200",
    "RSI", "Momentum", "ROC_12",
    "MACD", "MACD_Signal",
    "ATR_14", "ADX_14",
    "STOCH_K", "STOCH_D",
    "CCI_20", "VWAP_20", "VOL_SMA20",
    # Bollinger / volatility context
    "BB_PctB", "BB_Width",
    # Volume structure
    "OBV", "OBV_Slope",
    # Price context vs rolling extremes
    "Dist_From_High", "Dist_From_Low",
    # Standardization and candle geometry
    "Z_Score", "Body_Size", "Upper_Shadow", "Lower_Shadow",
    # Time features
    "Day_Of_Week", "Day_Of_Month",
    # Lagged features and differences
    "Close_Lag1", "Close_Diff",
    "RSI_Lag1", "RSI_Diff",
    "Volume_Lag1", "Volume_Diff",
    "OBV_Lag1", "OBV_Diff",

    # Smart Indicators (Stacking)
    "feat_rsi_signal", "feat_rsi_acc",
    "feat_ema_signal", "feat_ema_acc",
    "feat_bb_signal", "feat_bb_acc",

    # Market Context
    "feat_mkt_trend", "feat_mkt_volatility", "feat_rel_strength",

    # Fundamentals
    "marketCap", "peRatio", "eps", "dividendYield", "sector", "industry",
]

# Mapping technical feature names to human-readable explanations.
# Used for Phase 8: Explainable AI.
FEATURE_EXPLANATIONS = {
    "RSI": "Relative Strength Index (Momentum)",
    "MACD": "MACD Trend Divergence",
    "SMA_50": "50-Day Moving Average Support",
    "SMA_200": "200-Day Institutional Trend",
    "EMA_50": "50-Day Exponential Trend",
    "EMA_200": "200-Day Institutional EMA",
    "OBV": "On-Balance Volume (Flow)",
    "OBV_Slope": "Volume Accumulation Intensity",
    "VWAP_20": "Volume Weighted Average Price",
    "Momentum": "Price Velocity & Trend Strength",
    "ROC_12": "Rate of Change (12-period)",
    "ADX_14": "Trend Strength Index",
    "STOCH_K": "Stochastic Oscillator K",
    "BB_PctB": "Bollinger Bands Relative Position",
    "BB_Width": "Volatility Breakout Potential",
    "Close_Diff": "Daily Price Change Intensity",
    "RSI_Diff": "RSI Acceleration/Deceleration",
    "Z_Score": "Statistical Price Deviation",
    "Body_Size": "Candle Stick Pattern Strength",
    "Upper_Shadow": "Selling Pressure Intensity",
    "Lower_Shadow": "Buying Pressure Support",
    "VOL_SMA20": "Volume Trend vs Average",
    "Dist_From_High": "Proximity to Recent Highs",
    "Dist_From_Low": "Proximity to Recent Lows",
    "marketCap": "Company Market Valuation",
    "peRatio": "Price to Earnings (Valuation)",
    "eps": "Earnings Per Share (Profitability)",
    "dividendYield": "Dividend Payment Strength",
    "sector": "Economic Sector Context",
    "industry": "Specific Industry Dynamics",
}

def get_top_reasons(model: Any, predictors: List[str], limit: int = 3) -> List[str]:
    """Extract top N human-readable reasons from model feature importance."""
    top_reasons = []
    try:
        importances = []
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "get_feature_importance"):
            importances = model.get_feature_importance()
        
        if len(importances) > 0:
            # Pair predictors with their importance
            feat_imp = list(zip(predictors, importances))
            # Sort by importance descending
            # LightGBM might return more importances than predictors if not careful, 
            # but usually they match the length of feature_name()
            feat_imp.sort(key=lambda x: x[1], reverse=True)
            # Take top N and map to readable names
            for f_name, imp in feat_imp[:limit]:
                reason = FEATURE_EXPLANATIONS.get(f_name, f_name)
                top_reasons.append(reason)
    except Exception as e:
        print(f"Error extracting feature importance: {e}")
    return top_reasons


# Standard predictors for default RandomForest (Legacy)
RF_PREDICTORS = ["Close", "Volume", "SMA_50", "SMA_200", "RSI", "Momentum"]

def _init_supabase():
    global supabase
    if supabase is None:
        url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not key:
            key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        # print(f"DEBUG: Init Supabase. URL={url is not None}, KEY={key is not None}")
        if url and key:
            try:
                supabase = create_client(url, key)
                print("DEBUG: Supabase client initialized successfully")
            except Exception as e:
                print(f"Failed to init Supabase: {e}")
        else:
            print(f"DEBUG: Supabase env vars missing. URL={url}, KEY={'HIDDEN' if key else 'None'}")


# Define absolute paths for reliability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _finite_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _sanitize_fundamentals(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (fundamentals or {}).items():
        if isinstance(v, (int, float, np.number)):
            fv = _finite_float(v)
            out[k] = fv if fv is not None else None
        else:
            out[k] = v
    return out


def _last_trading_day(today: dt.date) -> dt.date:
    # Very small heuristic (no exchange calendar):
    # - If weekend, roll back to Friday
    if today.weekday() == 5:
        return today - dt.timedelta(days=1)
    if today.weekday() == 6:
        return today - dt.timedelta(days=2)
    return today




def _safe_cache_key(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_")


def _infer_symbol_exchange(ticker: str, exchange_hint: Optional[str] = None) -> Tuple[str, str]:
    """
    Standardize symbol/exchange inference for Supabase lookups.
    Returns (symbol_clean, exchange_clean)
    Example: 'AAPL.US' -> ('AAPL', 'US')
    Example: 'COMI.CC' -> ('COMI', 'EGX')
    """
    t = ticker.strip().upper()
    
    # Mapping table for country names to exchange codes
    country_to_ex = {
        "egypt": "EGX",
        "egx": "EGX",
        "ca": "EGX",
        "cc": "EGX",
        "cairo": "EGX",
        "usa": "US",
        "us": "US",
        "argentina": "BA",
        "brazil": "SA",
        "south korea": "KQ",
        "canada": "TO",
        "uk": "LSE",
        "france": "PA",
        "germany": "F"
    }

    # 1. Split by dot if present
    if "." in t:
        parts = t.split(".")
        s = parts[0]
        e = parts[1]
        # Map known variations
        if e.lower() in country_to_ex:
            e = country_to_ex[e.lower()]
        return s, e
        
    # 2. Use hint if provided
    if exchange_hint:
        e = exchange_hint.strip().lower()
        if e in country_to_ex:
            e = country_to_ex[e]
        else:
            e = e.upper()
        return t, e
        
    # 3. Default fallback (guessing) - mostly US or EGX in this app context
    # If it's 4+ letters and no dots, often US. If 3 letters, could be either.
    # But usually the ticker should have the suffix if it's EGX (like COMI.EGX).
    return t, "US"


def check_local_cache(symbol: str, exchange: Optional[str] = None) -> bool:
    """Checks if data exists in Supabase."""
    # Fast path: bulk cache
    try:
        s, e = _infer_symbol_exchange(symbol, exchange)
        bulk = _get_exchange_bulk_data(e)
        if s.upper() in bulk:
            return True
    except Exception:
        pass

    # 1. Check Supabase First
    _init_supabase()
    if supabase:
        try:
            s, e = _infer_symbol_exchange(symbol, exchange)
            res = supabase.table("stock_prices").select("count", count="exact").eq("symbol", s).eq("exchange", e).limit(1).execute()
            if res.count and res.count > 0:
                return True
        except Exception as ex:
            print(f"DEBUG: Supabase count check failed: {ex}")

    return False


_CACHED_TICKERS_SET = None
_CACHED_TICKERS_TS = 0
_CACHE_TTL_SECONDS = 30

_EXCHANGE_BULK_CACHE: Dict[str, Dict[str, Any]] = {}
_EXCHANGE_BULK_TTL_SECONDS = 900  # 15 minutes; adjust as needed
_BULK_LOAD_LOCKS: Dict[str, threading.Lock] = {}
_GLOBAL_LOAD_LOCK = threading.Lock()

# ----------------------
# Model Cache (avoid reloading per symbol)
# ----------------------
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
_MODEL_CACHE_TTL_SECONDS = 1800  # 30 minutes; adjust as needed


# ----------------------
# Indicator Cache (avoid recomputing TA each scan)
# ----------------------
class IndicatorCache:
    def __init__(self, max_age_seconds: int = 86400):  # 24h
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_age = max_age_seconds
        self.lock = Lock()

    def _cache_key(self, symbol: str, exchange: str) -> str:
        return f"{exchange.upper()}_{symbol.upper()}"

    def _data_hash(self, df: pd.DataFrame) -> str:
        # Hash length and last close/volume to detect data changes
        if df.empty: return "empty"
        
        last_row = df.iloc[-1]
        close = last_row.get("close", last_row.get("Close", 0))
        vol = last_row.get("volume", last_row.get("Volume", 0))
        
        data_str = f"{len(df)}_{close}_{vol}"
        return hashlib.md5(data_str.encode()).hexdigest()[:12]

    def get_with_indicators(self, symbol: str, exchange: str, df: pd.DataFrame, indicator_func) -> pd.DataFrame:
        cache_key = f"IND_{self._cache_key(symbol, exchange)}"
        data_hash = self._data_hash(df)

        with self.lock:
            entry = self.cache.get(cache_key)
            if entry and (time.time() - entry["ts"] < self.max_age) and entry.get("data_hash") == data_hash:
                return entry["data"].copy()

        df_with_ind = indicator_func(df)

        with self.lock:
            self.cache[cache_key] = {
                "data": df_with_ind.copy(),
                "data_hash": data_hash,
                "ts": time.time(),
            }
        return df_with_ind

    def get_with_massive_features(self, symbol: str, exchange: str, df: pd.DataFrame) -> pd.DataFrame:
        from train_exchange_model import add_massive_features
        cache_key = f"MASS_{self._cache_key(symbol, exchange)}"
        data_hash = self._data_hash(df)

        with self.lock:
            entry = self.cache.get(cache_key)
            if entry and (time.time() - entry["ts"] < self.max_age) and entry.get("data_hash") == data_hash:
                return entry["data"].copy()

        df_with_feat = add_massive_features(df)

        with self.lock:
            self.cache[cache_key] = {
                "data": df_with_feat.copy(),
                "data_hash": data_hash,
                "ts": time.time(),
            }
        return df_with_feat

    def clear(self, symbol: Optional[str] = None, exchange: Optional[str] = None):
        with self.lock:
            if symbol and exchange:
                key = self._cache_key(symbol, exchange)
                if key in self.cache:
                    del self.cache[key]
            else:
                self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            return {"cached_symbols": len(self.cache), "max_age_seconds": self.max_age}


_INDICATOR_CACHE: Optional[IndicatorCache] = None


def _get_indicator_cache(max_age_seconds: int = 86400) -> IndicatorCache:
    global _INDICATOR_CACHE
    if _INDICATOR_CACHE is None:
        _INDICATOR_CACHE = IndicatorCache(max_age_seconds=max_age_seconds)
    return _INDICATOR_CACHE


def _get_data_with_indicators_cached(symbol: str, exchange: str, df: pd.DataFrame, indicator_func) -> pd.DataFrame:
    cache = _get_indicator_cache()
    return cache.get_with_indicators(symbol, exchange, df, indicator_func)


def _get_exchange_bulk_data(exchange: str, from_date: Optional[str] = None, to_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all price data for an exchange in a single Supabase query (paginated) and cache it.
    Returns mapping symbol -> DataFrame indexed by date.
    """
    if not exchange:
        return {}

    now = time.time()
    cache_key = f"{exchange.upper()}_{from_date or 'ALL'}_{to_date or 'NOW'}"

    # 1. Simple cache check
    cached = _EXCHANGE_BULK_CACHE.get(cache_key)
    if cached and (now - cached.get("ts", 0) < _EXCHANGE_BULK_TTL_SECONDS):
        print(f"DEBUG: Bulk Cache Hit for {cache_key}")
        return cached.get("data", {})

    # 2. Get or create a lock for this specific cache key to prevent "loop refetch"
    with _GLOBAL_LOAD_LOCK:
        if cache_key not in _BULK_LOAD_LOCKS:
            _BULK_LOAD_LOCKS[cache_key] = Lock()
        lock = _BULK_LOAD_LOCKS[cache_key]

    with lock:
        # Re-check cache after acquiring lock
        cached = _EXCHANGE_BULK_CACHE.get(cache_key)
        if cached and (time.time() - cached.get("ts", 0) < _EXCHANGE_BULK_TTL_SECONDS):
            print(f"DEBUG: Bulk Cache Hit (locked) for {cache_key}")
            return cached.get("data", {})

        _init_supabase()
        if not supabase:
            return {}

        print(f"DEBUG: Starting bulk load for {cache_key} (from_date={from_date})")
        start_time = time.time()

        try:
            # 1. Build base query with exact count request in select
            page_size = 1000 # Supabase/PostgREST default limit
            query = supabase.table("stock_prices") \
                .select("symbol,exchange,date,open,high,low,close,volume", count="exact") \
                .eq("exchange", exchange.upper())
            
            if from_date: query = query.gte("date", from_date)
            if to_date: query = query.lte("date", to_date)
            query = query.order("symbol", desc=False).order("date", desc=False)

            # 2. Fetch first page + total count
            fetch_start = time.time()
            res0 = query.range(0, page_size - 1).execute()
            all_rows = res0.data or []
            total_count = res0.count or len(all_rows)
            
            print(f"DEBUG: Starting Parallel Bulk Load for {exchange.upper()} ({total_count} total rows)")

            if total_count > page_size:
                offsets = range(page_size, total_count, page_size)
                
                def _fetch_page(off, retries=5): # Increased retries
                    for attempt in range(retries):
                        try:
                            # Re-build query to avoid any shared state issues
                            r = supabase.table("stock_prices") \
                                .select("symbol,exchange,date,open,high,low,close,volume") \
                                .eq("exchange", exchange.upper())
                            if from_date: r = r.gte("date", from_date)
                            if to_date: r = r.lte("date", to_date)
                            r = r.order("symbol", desc=False).order("date", desc=False)
                            
                            res = r.range(off, off + page_size - 1).execute()
                            return res.data or []
                        except Exception as e:
                            wait = (attempt + 1) * 3
                            if attempt > 1: # Only log after first failure to reduce noise
                                print(f"DEBUG: Retry {attempt+1}/{retries} for offset {off} due to: {e}. Waiting {wait}s...")
                            time.sleep(wait)
                    print(f"ERROR: Failed to fetch page at offset {off} after {retries} attempts.")
                    return []

                # Fetch remaining pages in parallel (Conservative 3 workers)
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(_fetch_page, o) for o in offsets]
                    for f in as_completed(futures):
                        all_rows.extend(f.result())

            fetch_duration = time.time() - fetch_start
            print(f"DEBUG: Supabase Parallel Fetch complete. Took {fetch_duration:.2f}s for {len(all_rows)} rows across {len(set(r['symbol'] for r in all_rows))} symbols")

            if not all_rows:
                return {}

            process_start = time.time()
            df_all = pd.DataFrame(all_rows)
            if df_all.empty:
                return {}

            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
            df_all = df_all.dropna(subset=["date"]).sort_values(["symbol", "date"])

            data_by_symbol: Dict[str, pd.DataFrame] = {}
            for sym, grp in df_all.groupby("symbol"):
                g = grp.set_index("date")["open high low close volume".split()]
                g = g.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
                data_by_symbol[str(sym).upper()] = g

            _EXCHANGE_BULK_CACHE[cache_key] = {"ts": now, "data": data_by_symbol, "rows": len(df_all)}
            total_duration = time.time() - start_time
            print(f"DEBUG: Bulk cached {len(df_all)} rows for {cache_key} ({len(data_by_symbol)} symbols) in {total_duration:.2f}s (Processing: {time.time()-process_start:.2f}s)")
            return data_by_symbol
        except Exception as e:
            print(f"DEBUG: Bulk load failed for exchange {cache_key}: {e}")
            return {}


def _get_model_cached(model_path: str):
    now = time.time()
    entry = _MODEL_CACHE.get(model_path)
    if entry and (now - entry.get("ts", 0) < _MODEL_CACHE_TTL_SECONDS):
        return entry.get("model"), entry.get("predictors"), entry.get("is_lgbm_artifact", False)
    return None


def _set_model_cache(model_path: str, model: Any, predictors: Optional[List[str]] = None, is_lgbm_artifact: bool = False) -> None:
    _MODEL_CACHE[model_path] = {
        "model": model,
        "predictors": predictors,
        "is_lgbm_artifact": is_lgbm_artifact,
        "ts": time.time(),
    }

def get_cached_tickers() -> set:
    """Returns a set of all ticker names. Prioritizes Supabase. Cached for 30s."""
    global _CACHED_TICKERS_SET, _CACHED_TICKERS_TS
    
    import time
    now = time.time()
    
    if _CACHED_TICKERS_SET is not None and (now - _CACHED_TICKERS_TS) < _CACHE_TTL_SECONDS:
        return _CACHED_TICKERS_SET

    found = set()

    # 1. Try Supabase
    _init_supabase()
    if supabase:
        try:
            # Use high-performance RPC to get all active symbols (with prices, no funds)
            res = supabase.rpc("get_active_symbols").execute()
            if res.data:
                for row in res.data:
                    found.add(f"{row['symbol']}.{row['exchange']}")
        except Exception as e:
            print(f"Error fetching cached tickers RPC: {e}")
        
    _CACHED_TICKERS_SET = found
    _CACHED_TICKERS_TS = now
    return found

def is_ticker_synced(symbol: str, exchange: Optional[str] = None) -> bool:
    """Checks if a symbol.exchange combination exists in our cloud database."""
    cached_set = get_cached_tickers()
    s_upper = _safe_cache_key(symbol)
    
    # Try exact match first
    if s_upper in cached_set: return True
    
    # Try with inferred/standardized suffix if exchange is provided
    if exchange:
        e_upper = exchange.upper()
        if e_upper in ["CC", "CA"]: e_upper = "EGX"
        mapping = {"EGX": "EGX", "US": "US", "NYSE": "US", "NASDAQ": "US"}
        suffix = mapping.get(e_upper, e_upper)
        
        # Strip existing suffix if any to avoid DOUBLE suffixes
        base = s_upper.split('.')[0]
        if f"{base}.{suffix}" in cached_set: return True
        
    return False

def get_supabase_inventory() -> List[Dict[str, Any]]:
    """Call get_inventory_stats RPC and group by country for frontend use."""
    _init_supabase()
    if not supabase: return []

    from api.symbols_local import load_country_summary
    local_summary = load_country_summary()
    expected_map = {}
    for country, data in local_summary.items():
        exchanges = data.get("Exchanges", {})
        for ex, count in exchanges.items():
            expected_map[ex] = {"count": count, "country": country}

    first_attempt_success = False
    stats = []

    # 1. Try stats from RPC (Primary)
    try:
        res = supabase.rpc("get_inventory_stats").execute()
        if res.data:
            stats = res.data
            first_attempt_success = True
    except Exception as e:
        print(f"Warning: Primary inventory stats RPC failed: {e}. Falling back to Fundamentals + Local stats.")

    # 2. Get exchange-to-country mapping from fundamentals if possible (Shared)
    mapping = {}
    try:
        # Optimization: Only count symbols per country to find the 'primary' country for an exchange
        # Actually, the expected_map from local summary is the most reliable.
        # We only use fundamentals as secondary.
        meta_res = supabase.table("stock_fundamentals").select("exchange, country:data->>country").execute()
        if meta_res.data:
            # Simple voting logic: count symbols per (exchange, country)
            votes = {} 
            for row in meta_res.data:
                ex = row.get('exchange')
                c = row.get('country')
                if ex and c:
                    key = (ex, c)
                    votes[key] = votes.get(key, 0) + 1
            
            # Pick highest vote for each exchange
            ex_best = {}
            for (ex, c), count in votes.items():
                if ex not in ex_best or count > ex_best[ex][1]:
                    ex_best[ex] = (c, count)
            
            for ex, (c, _) in ex_best.items():
                mapping[ex] = c
    except Exception as e:
        print(f"Warning: Fundamentals metadata fetch failed: {e}")

    # 3. If primary RPC failed, build stats from fundamentals aggregation (Fallback)
    if not stats:
        # We can't easily get price_count without the heavy query, so we set it to 0 or -1 (unknown)
        # But we can get accurate fund_count from stock_fundamentals
        try:
            # Group by exchange manually from the meta_res we just got, or fetch if needed
            fund_counts = {}
            if meta_res.data:
                for row in meta_res.data:
                    ex = row.get('exchange')
                    if ex:
                        fund_counts[ex] = fund_counts.get(ex, 0) + 1
            
            # Construct a partial stats list
            for ex, count in fund_counts.items():
                stats.append({
                    "exchange": ex,
                    "price_count": 0, # Cannot determine efficiently on timeout
                    "fund_count": count,
                    "last_update": None
                })
        except Exception as e:
            print(f"Warning: Fallback stats generation failed: {e}")


    try:
        # 4. Join and group (Shared Logic)
        out = []
        mapped_exchanges = set()

        for row in stats:
            ex = row.get('exchange', 'Unknown')
            expected = expected_map.get(ex, {})
            # Enrich row
            row['country'] = mapping.get(ex) or expected.get("country", "Unknown")
            
            # Dynamic expected count correction
            country_name = row['country']
            if country_name != "Unknown":
                try:
                    from api.symbols_local import load_symbols_for_country
                    syms = load_symbols_for_country(country_name)
                    # Count for this specific exchange
                    actual_count = sum(1 for s in syms if str(s.get("Exchange", s.get("exchange", ""))).upper() == ex.upper())
                    if actual_count > 0:
                        expected["count"] = actual_count
                except:
                    pass

            row['expected_count'] = expected.get("count", 0)
            # Add camelCase aliases for the UI if needed
            row['priceCount'] = row.get('price_count', 0)
            row['fundCount'] = row.get('fund_count', 0)
            row['expectedCount'] = row['expected_count']
            row['lastUpdate'] = row.get('last_update')
            
            # If we are in fallback mode (price_count is 0 but we know we have some data potentially),
            # we might want to flag it or leave it as 0. 
            # For now 0 is fine, UI shows "0 / N".
            
            out.append(row)
            mapped_exchanges.add(ex)

        # 5. Add missing exchanges from local summary
        for ex, expected in expected_map.items():
            if ex not in mapped_exchanges:
                out.append({
                    "exchange": ex,
                    "country": expected["country"],
                    "price_count": 0,
                    "fund_count": 0,
                    "expected_count": expected["count"],
                    "priceCount": 0,
                    "fundCount": 0,
                    "expectedCount": expected["count"],
                    "last_update": None,
                    "lastUpdate": None
                })
            
        return out
    except Exception as e:
        print(f"Error processing inventory stats: {e}")
    return []


def get_supabase_countries() -> List[str]:
    """Fetch unique countries from Supabase fundamentals that have price data."""
    _init_supabase()
    if not supabase: return []
    try:
        res = supabase.rpc("get_active_countries").execute()
        if res.data:
            return [r['country'] for r in res.data]
    except Exception as e:
        print(f"Error fetching active countries RPC: {e}")
    return []


def get_supabase_symbols(country: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch symbols from Supabase that have price data, ensuring high coverage."""
    _init_supabase()
    if not supabase: return []
    
    try:
        # 1. Try RPC first (active symbols with metadata)
        params = {}
        if country:
            params['p_country'] = country
        
        rpc_res = supabase.rpc("get_active_symbols", params).execute()
        symbols_map = {}
        
        if rpc_res.data:
            for r in rpc_res.data:
                symbols_map[r['symbol']] = {
                    "symbol": r['symbol'],
                    "exchange": r['exchange'],
                    "name": r.get('name', ''),
                    "country": r.get('country', country or ''),
                    "hasLocal": True
                }

        # 2. Add fallback for symbols in stock_fundamentals
        try:
            fund_query = supabase.table("stock_fundamentals").select("symbol, exchange, country:data->>country")
            if country:
                # Some symbols might have nested data or flat country field
                fund_query = fund_query.eq("data->>country", country)
            
            fund_res = fund_query.execute()
            if fund_res.data:
                for r in fund_res.data:
                    sym = r['symbol']
                    if sym not in symbols_map:
                        symbols_map[sym] = {
                            "symbol": sym,
                            "exchange": r['exchange'],
                            "name": "", # Basic
                            "country": r.get('country', country or 'Unknown'),
                            "hasLocal": True
                        }
        except Exception as fund_err:
            print(f"Fallback fundamentals query failed: {fund_err}")

        # 3. Last fallback: stock_prices table (limited)
        try:
            # Map countries to exchanges for direct querying if we have the info
            country_to_ex = {
                "Brazil": "SA",
                "South Korea": "KQ",
                "Egypt": "EGX",
                "USA": "US"
            }
            
            ex_filter = country_to_ex.get(country) if country else None
            
            # Use a slightly larger range if needed
            # To get more unique symbols, we fetch a few pages or a larger block
            # Since we can't do DISTINCT easily, we fetch 2000 recent price records
            price_query = supabase.table("stock_prices").select("symbol, exchange")
            if ex_filter:
                price_query = price_query.eq("exchange", ex_filter)
            
            # Fetch in two chunks to avoid PostgREST 1000 limit issues
            p1 = price_query.range(0, 999).execute()
            p2 = price_query.range(1000, 1999).execute()
            
            all_price_rows = (p1.data or []) + (p2.data or [])
            
            if all_price_rows:
                for r in all_price_rows:
                    sym = r['symbol']
                    if sym not in symbols_map:
                        symbols_map[sym] = {
                            "symbol": sym,
                            "exchange": r['exchange'],
                            "name": f"Unknown ({sym})",
                            "country": country or 'Unknown',
                            "hasLocal": True
                        }
        except Exception as price_err:
            print(f"Fallback price query failed: {price_err}")

        return sorted(list(symbols_map.values()), key=lambda x: x['symbol'])
        
    except Exception as e:
        print(f"Error fetching symbols: {e}")
    return []


def _candidate_symbols(ticker: str, exchange: Optional[str] = None) -> List[str]:
    t = ticker.strip().upper()
    if "." in t:
        return [t]
    
    # Mapping table for common exchange names in JSON to EODHD suffixes
    mapping = {
        "EGX": "EGX",   # Match your local cache files
        "US": "US",    # USA
        "NYSE": "US",
        "NASDAQ": "US",
        "LSE": "LSE",  # UK
        "WAR": "WAR",  # Poland
        "TO": "TO",    # Canada
        "V": "V",      # Canada Venture
        "PA": "PA",    # France
        "F": "F",      # Germany
    }
    
    candidates = []
    if exchange and exchange.upper() in mapping:
        suffix = mapping[exchange.upper()]
        candidates.append(f"{t}.{suffix}")
        candidates.append(t)
    else:
        # Fallbacks only when exchange is unknown/not provided
        candidates.append(f"{t}.US")
        candidates.append(t)
    
    # Remove duplicates while preserving order
    unique_candidates = []
    for c in candidates:
        if c not in unique_candidates:
            unique_candidates.append(c)
    return unique_candidates


def _normalize_eodhd_eod_result(raw: Union[pd.DataFrame, List[Dict[str, Any]], Any]) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame()

    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    elif isinstance(raw, list):
        df = pd.DataFrame(raw)
    else:
        # Best-effort fallback
        try:
            df = pd.DataFrame(raw)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return df

    # EODHD commonly includes a "date" field.
    # If it exists, use it as the datetime index.
    date_col = None
    for cand in ("date", "Date"):
        if cand in df.columns:
            date_col = cand
            break

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

    return df


def get_stock_data_eodhd(
    api: APIClient,
    ticker: str,
    from_date: str,
    to_date: Optional[str] = None,
    tolerance_days: int = 0,
    exchange: Optional[str] = None,
    force_local: bool = False,
) -> pd.DataFrame:
    
    possible_names = [ticker]
    
    # 0. Try bulk cache for the whole exchange first (single query, cached)
    try:
        s, e = _infer_symbol_exchange(ticker, exchange)
        bulk = _get_exchange_bulk_data(e, from_date, to_date)
        if s.upper() in bulk:
            df_cached = bulk[s.upper()]
            if not df_cached.empty:
                df = df_cached.copy()
                if from_date:
                    df = df[df.index >= pd.to_datetime(from_date)]
                if to_date:
                    df = df[df.index <= pd.to_datetime(to_date)]
                if not df.empty:
                    print(f"DEBUG: Bulk cache hit for {s}.{e} ({len(df)} rows)")
                    return df
    except Exception as ex:
        print(f"DEBUG: Bulk cache lookup failed for {ticker}: {ex}")

    # 1. Try Supabase per-symbol (paginated)
    _init_supabase()
    if supabase:
        try:
            s, e = _infer_symbol_exchange(ticker, exchange)
            
            # Supabase has a default limit of 1000 rows, so we need to paginate
            # to get all historical data for stocks with long history
            all_data = []
            page_size = 1000
            offset = 0
            
            while True:
                res = (
                    supabase.table("stock_prices")
                    .select("date,open,high,low,close,volume")
                    .eq("symbol", s)
                    .eq("exchange", e)
                    .order("date", desc=False)
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                
                if not res.data:
                    break
                    
                all_data.extend(res.data)
                
                # If we got fewer than page_size, we've reached the end
                if len(res.data) < page_size:
                    break
                    
                offset += page_size
            
            if all_data:
                df = pd.DataFrame(all_data)
                if not df.empty:
                    # Convert to standard format
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index()
                    # Filter by date range
                    if from_date:
                        df = df[df.index >= pd.to_datetime(from_date)]
                    if to_date:
                        df = df[df.index <= pd.to_datetime(to_date)]
                    if not df.empty:
                        print(f"DEBUG: Supabase hit for {s}.{e} ({len(df)} rows)")
                        return df
        except Exception as e:
            print(f"Supabase read error for {ticker}: {e}")

    if force_local:
        return pd.DataFrame() # Return empty instead of raising error if cloud-only

    # No cloud data, try API
    try:
        df = api.get_eod_historical_stock_market_data(
            symbol=ticker,
            period="d",
            order="a",
            from_date=from_date,
        )
        
        # Handle 401/Unauthorized returned as plain string by the library
        if isinstance(df, str) and ("Unauthorized" in df or "401" in df):
             if best_stale_df is not None: return best_stale_df
             raise ValueError("EODHD API Key Unauthorized (401)")

        df = _normalize_eodhd_eod_result(df)

        if df is None or df.empty:
            raise ValueError(f"No historical data returned for {ticker}")

        # Automatic Sync (Directly to Supabase)
        sync_df_to_supabase(ticker, df)
        return df
    except Exception as e:
        # If any error happens and we had some data (though we returned it above, 
        # this is a safety net), return it.
        raise e



def get_stock_data_yahoo(
    ticker: str,
    from_date: str = "2020-01-01",
    force_local: bool = False
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.
    """
    # Disk-less: No local file check. 
    # Logic: Always fetch from Yahoo if get_stock_data-Supabase failed.
    if force_local:
        return pd.DataFrame()

    # 2. Fetch from Yahoo
    # Yahoo Ticker Normalization
    yf_ticker = ticker
    if ticker.endswith(".US"):
        yf_ticker = ticker.replace(".US", "")
    elif ticker.endswith(".EGX"):
        # Yahoo often uses .CA for Egypt (Cairo)
        base = ticker.replace(".EGX", "")
        yf_ticker = f"{base}.CA"
    
    try:
        # Download history
        df = yf.download(yf_ticker, start=from_date, progress=False, auto_adjust=True)
        
        # Yahoo returns MultiIndex columns sometimes if multiple tickers (not here)
        # normalize columns: Open, High, Low, Close, Volume
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)
        
        # Standardize names
        df = df.rename(columns={
            "Open": "open", 
            "High": "high", 
            "Low": "low", 
            "Close": "close", 
            "Volume": "volume"
        })
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        if df.empty:
             raise ValueError(f"No data found for {yf_ticker} on Yahoo")

        # Sync Directly
        sync_df_to_supabase(ticker, df)
        return df
        
    except Exception as e:
        raise ValueError(f"Yahoo fetch failed for {yf_ticker}: {e}")


def _get_supabase_info(ticker: str) -> Dict[str, Any]:
    """Helper to find the latest available date and record count for a ticker in Supabase."""
    _init_supabase()
    out = {"last_date": None, "count": 0}
    if not supabase:
        return out
    try:
        sb_symbol = ticker
        sb_exchange = "US"
        if "." in ticker:
            parts = ticker.split(".")
            sb_symbol = parts[0]
            sb_exchange = parts[1]
            if sb_exchange in ["CC", "CA"]: sb_exchange = "EGX"
            
        res = supabase.table("stock_prices")\
            .select("date", count="exact")\
            .eq("symbol", sb_symbol)\
            .eq("exchange", sb_exchange)\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        if res.data:
            out["last_date"] = pd.to_datetime(res.data[0]["date"]).date()
            out["count"] = res.count or 0
    except Exception as e:
        print(f"Error checking Supabase info for {ticker}: {e}")
    return out

def _get_supabase_last_date(ticker: str) -> Optional[dt.date]:
    return _get_supabase_info(ticker)["last_date"]


def update_stock_data(
    api: APIClient,
    ticker: str,
    source: str = "eodhd",
    max_days: int = 365
) -> Tuple[bool, str]:
    """
    Updates stock data for a given ticker and upserts directly to Supabase.
    source: 'eodhd'
    """
    print(f"DEBUG IN UPDATE: Ticker: {ticker}, Source received: {source}")
    if source.lower() != "eodhd":
        return False, "Only EODHD is supported for price updates"
    if api is None:
        return False, "EODHD API client is not configured"

    today = dt.date.today()
    
    # Disk-less: Check Supabase for last date and count
    info = _get_supabase_info(ticker)
    last_date = info["last_date"]
    current_count = info["count"]
    
    is_up_to_date = last_date and last_date >= _last_trading_day(today)
    has_enough_history = current_count >= max_days
    
    print(f"DEBUG SYNC: {ticker} | last_date={last_date} | count={current_count} | max_days={max_days} | up_to_date={is_up_to_date} | enough_history={has_enough_history}")
    
    if is_up_to_date and has_enough_history:
         return True, "Already up to date and sufficient history in Cloud"

    # Determine if we need to fetch anything
    # Logic: If NOT up-to-date OR NOT enough history, we fetch.
    
    # Needs backfill if history is too short
    needs_backfill = not has_enough_history

    # EODHD Logic
    try:
        if needs_backfill:
            # Force a full historical download to get enough history
            print(f"BACKFILL/FULL SYNC MODE: {ticker} has {current_count} records, need {max_days}. Downloading full history.")
            # We fetch at least max_days, but ensure we also get recent data by fetching up to today
            start_date = today - dt.timedelta(days=max_days + 120)  # Safe buffer
            df = api.get_eod_historical_stock_market_data(
                symbol=ticker, period="d", order="a", from_date=str(start_date)
            )
            df = _normalize_eodhd_eod_result(df)
            if df is None or df.empty:
                return False, "No historical data available (EODHD Full/Backfill)"
            
            sync_df_to_supabase(ticker, df)
            return True, f"Full sync/Backfill: {len(df)} rows to Cloud (was {current_count})"
        
        elif not is_up_to_date:
            # Just append new data
            start_date = (last_date + dt.timedelta(days=1)) if last_date else (today - dt.timedelta(days=max_days + 30))
            print(f"APPEND MODE: {ticker} syncing since {start_date} to today.")
            new_df = api.get_eod_historical_stock_market_data(
                symbol=ticker, period="d", order="a", from_date=str(start_date)
            )
            new_df = _normalize_eodhd_eod_result(new_df)
            
            if new_df is None or new_df.empty:
                return True, "No new data returned from API (EODHD Append)"
            
            sync_df_to_supabase(ticker, new_df)
            return True, f"Appended {len(new_df)} rows to Cloud"
        
        return True, "Already up to date"

    except Exception as e:
        return False, f"EODHD Error: {str(e)}"



def sync_df_to_supabase(ticker: str, df: pd.DataFrame) -> tuple[bool, str]:
    """Upsert a Pandas DataFrame of stock prices directly to Supabase."""
    _init_supabase()
    if not supabase:
        return False, "Supabase not initialized"
    if df.empty:
        return False, "DataFrame is empty"
        
    try:
        # Normalize columns to lowercase for case-insensitive access
        df.columns = [c.lower() for c in df.columns]
        
        # Reset index if date is index
        if df.index.name and df.index.name.lower() == 'date':
            df = df.reset_index()
        elif not 'date' in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or 'index': 'date'})

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date']).sort_values('date').tail(5000) # Increased buffer from 1000
                
            rows = []
            sb_symbol = ticker
            sb_exchange = "US"
            if "." in ticker:
                parts = ticker.split(".")
                sb_symbol = parts[0]
                sb_exchange = parts[1]
                if sb_exchange in ["CC", "CA"]: sb_exchange = "EGX"
            
            for _, row in df.iterrows():
                adj_close = row.get('adjusted_close')
                if adj_close is None:
                    adj_close = row.get('close')

                rows.append({
                    "symbol": sb_symbol,
                    "exchange": sb_exchange,
                    "date": row['date'].strftime('%Y-%m-%d'),
                    "open": _finite_float(row.get('open')),
                    "high": _finite_float(row.get('high')),
                    "low": _finite_float(row.get('low')),
                    "close": _finite_float(row.get('close')),
                    "adjusted_close": _finite_float(adj_close),
                    "volume": int(row['volume']) if pd.notna(row.get('volume')) else None,
                })
            
            if rows:
                # Use chunks of 100 to avoid request size limits
                for i in range(0, len(rows), 100):
                    supabase.table("stock_prices").upsert(rows[i:i+100], on_conflict="symbol,exchange,date").execute()
                
                # Fetch final count and range for verification
                final_info = _get_supabase_info(ticker)
                min_date = df['date'].min().date()
                max_date = df['date'].max().date()
                
                msg = f"Synced {len(rows)} rows ({min_date} to {max_date}). Total in cloud: {final_info['count']}"
                print(f"DEBUG: {msg}")
                return True, msg
            else:
                return False, "No valid rows found to sync"
    except Exception as e:
        msg = f"Supabase sync error for {ticker}: {e}"
        print(msg)
        return False, msg
    
    return False, "Unknown error"


def sync_local_to_supabase(ticker: str, file_path: str) -> tuple[bool, str]:
    """Read local CSV and upsert to Supabase stock_prices."""
    _init_supabase()
    if not supabase:
        return False, "Supabase not initialized"
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
        
    try:
        saved_df = pd.read_csv(file_path)
        return sync_df_to_supabase(ticker, saved_df)
    except Exception as e:
        msg = f"Supabase sync error for {ticker}: {e}"
        print(msg)
        return False, msg
    
    return False, "Unknown error"



def sync_data_to_supabase(ticker: str, data: dict) -> tuple[bool, str]:
    """Upsert fundamental data directly to Supabase."""
    _init_supabase()
    if not supabase:
        return False, "Supabase not initialized"
    if not data:
        return False, "No data provided"
        
    try:
        sb_symbol = ticker
        sb_exchange = "US"
        if "." in ticker:
            parts = ticker.split(".")
            sb_symbol = parts[0]
            sb_exchange = parts[1]
            if sb_exchange in ["CC", "CA"]: sb_exchange = "EGX"
        
        # Clean symbol (remove fund_ prefix if present)
        # This handles the user's issue where "fund_ANFI" was stored instead of "ANFI"
        if sb_symbol.lower().startswith("fund_"):
            sb_symbol = sb_symbol[5:]  # Remove 'fund_' (5 chars)

        row = {
            "symbol": sb_symbol,
            "exchange": sb_exchange,
            "data": data,
            "updated_at": dt.datetime.utcnow().isoformat()
        }
        
        # Populate optional columns if available in data
        # Check for case-insensitive keys just in case
        clean_data = {k.lower(): v for k, v in data.items()}
        if "name" in clean_data:
            row["name"] = clean_data["name"]
        if "country" in clean_data:
            row["country"] = clean_data["country"]
        
        supabase.table("stock_fundamentals").upsert(row, on_conflict="symbol,exchange").execute()
        return True, "Fundamentals synced"
    except Exception as e:
        msg = f"Supabase fund sync error for {ticker}: {e}"
        print(msg)
        return False, msg


def clear_supabase_stock_prices() -> tuple[bool, str]:
    """Delete all records from Supabase stock_prices table."""
    _init_supabase()
    if not supabase:
        return False, "Supabase not initialized"
    
    try:
        # Using .neq("symbol", "") as a catch-all to delete all rows.
        res = supabase.table("stock_prices").delete().neq("symbol", "").execute()
        return True, "All stock prices cleared from Supabase"
    except Exception as e:
        msg = f"Error clearing stock prices: {e}"
        print(msg)
        return False, msg

def upsert_technical_indicators(
    symbol: str,
    exchange: str,
    date: str,
    close: float,
    volume: int,
    indicators: Dict[str, Any]
) -> tuple[bool, str]:
    """Upsert technical indicators to Supabase using RPC."""
    _init_supabase()
    if not supabase:
        return False, "Supabase not initialized"
    
    try:
        data = {
            "p_symbol": symbol,
            "p_exchange": exchange,
            "p_date": date,
            "p_close": close,
            "p_volume": int(volume),
            "p_ema_50": _finite_float(indicators.get("EMA_50")),
            "p_ema_200": _finite_float(indicators.get("EMA_200")),
            "p_sma_50": _finite_float(indicators.get("SMA_50")),
            "p_sma_200": _finite_float(indicators.get("SMA_200")),
            "p_rsi_14": _finite_float(indicators.get("RSI")),
            "p_macd": _finite_float(indicators.get("MACD")),
            "p_macd_signal": _finite_float(indicators.get("MACD_Signal")),
            "p_macd_histogram": _finite_float(indicators.get("MACD_Histogram")), # If implemented
            "p_adx_14": _finite_float(indicators.get("ADX_14")),
            "p_atr_14": _finite_float(indicators.get("ATR_14")),
            "p_bb_upper": _finite_float(indicators.get("BB_Upper")),
            "p_bb_lower": _finite_float(indicators.get("BB_Lower")),
            "p_stoch_k": _finite_float(indicators.get("STOCH_K")),
            "p_stoch_d": _finite_float(indicators.get("STOCH_D")),
            "p_roc_12": _finite_float(indicators.get("ROC_12")),
            "p_momentum_10": _finite_float(indicators.get("Momentum")),
            "p_vol_sma20": int(indicators.get("VOL_SMA20", 0)) if indicators.get("VOL_SMA20") else None,
            "p_vwap_20": _finite_float(indicators.get("VWAP_20")),
            "p_r_vol": _finite_float(indicators.get("R_VOL")),
            "p_change_pct": _finite_float(indicators.get("change_p"))
        }
        
        # Filter out keys with None to allow defaults in SQL if any
        params = {k: v for k, v in data.items() if v is not None}
        
        supabase.rpc("admin_upsert_technical_indicator", params).execute()
        return True, "Technical indicators upserted"
    except Exception as e:
        msg = f"Supabase tech upsert error for {symbol}: {e}"
        print(msg)
        return False, msg

            





def get_company_fundamentals(
    ticker: str,
    return_meta: bool = False,
    source: str = "auto",
) -> Any:
    # 0. Try Supabase
    _init_supabase()
    if supabase:
        try:
            s, e = _infer_symbol_exchange(ticker)
            res = supabase.table("stock_fundamentals").select("data").eq("symbol", s).eq("exchange", e).limit(1).execute()
            if res.data and res.data[0].get("data"):
                data = res.data[0]["data"]
                # Safety: Parse JSON if returned as string
                if isinstance(data, str):
                    try:
                        import json
                        data = json.loads(data)
                    except:
                        pass
                if return_meta:
                    return data, {"source": "supabase", "servedFrom": "supabase"}
                return data
        except Exception as e:
            print(f"Supabase fund read error: {e}")

    now_ts = int(dt.datetime.utcnow().timestamp())
    src = (source or "auto").lower()

    def _has_core_metrics(d: Dict[str, Any]) -> bool:
        if not isinstance(d, dict):
            return False
        core = ["marketCap", "peRatio", "eps", "dividendYield", "beta", "high52", "low52"]
        for k in core:
            if d.get(k) is not None:
                return True
        return False

    def _ret(data: Dict[str, Any], meta: Dict[str, Any]) -> Any:
        if return_meta:
            return data, meta
        return data

    def _try_tradingview() -> Any:
        upper = (ticker or "").strip().upper()
        if not upper: return None
        # Allow EGX/CC in TradingView as fallback
        # if upper.endswith(".EGX") or upper.endswith(".CC"): return None

        from tradingview_integration import fetch_tradingview_fundamentals_bulk
        bulk = fetch_tradingview_fundamentals_bulk([upper])
        if upper not in bulk: return None
        
        data, meta = bulk[upper]
        return _ret(data, {**meta, "servedFrom": "live_tradingview"})

    def _try_mubasher_egx() -> Any:
        upper = ticker.strip().upper()
        base_symbol = upper.split(".")[0]
        if not (upper.endswith(".EGX") or upper.endswith(".CC") or upper == base_symbol):
            return None

        try:
            import requests
            from bs4 import BeautifulSoup
        except Exception:
            return None

        url = f"https://english.mubasher.info/markets/EGX/stocks/{base_symbol}/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            overview_items = soup.find_all("div", class_="stock-overview__text-and-value-item")
            raw_map: Dict[str, str] = {}

            for item in overview_items:
                text_span = item.find("span", class_="stock-overview__text")
                value_span = item.find("span", class_="stock-overview__value")
                if text_span and value_span:
                    key = text_span.get_text(strip=True).replace(":", "").strip()
                    number_span = value_span.find("span", class_="number")
                    value = number_span.get_text(strip=True) if number_span else value_span.get_text(strip=True)
                    if key:
                        raw_map[key] = value

            def _parse_num(raw_val: str) -> Optional[float]:
                if raw_val is None:
                    return None
                t = str(raw_val).strip()
                if not t or t.lower() in {"n/a", "na", "-", "--"}:
                    return None
                t = t.replace(",", "").replace(" ", "")
                if t.endswith("%"): 
                    t = t[:-1]
                mult = 1.0
                if t.endswith("K"):
                    mult = 1e3
                    t = t[:-1]
                elif t.endswith("M"):
                    mult = 1e6
                    t = t[:-1]
                elif t.endswith("B"):
                    mult = 1e9
                    t = t[:-1]
                try:
                    v = float(t) * mult
                except Exception:
                    return None
                return v if np.isfinite(v) else None

            data: Dict[str, Any] = {
                "name": raw_map.get("Name") or raw_map.get("Company Name") or None,
                "country": "Egypt",
                "currency": "EGP",
            }

            for k, v in raw_map.items():
                lk = k.strip().lower()
                if "market cap" in lk:
                    data["marketCap"] = _parse_num(v)
                elif "p/e" in lk or lk in {"pe", "pe ratio"}:
                    data["peRatio"] = _parse_num(v)
                elif "eps" in lk:
                    data["eps"] = _parse_num(v)
                elif "dividend" in lk and "yield" in lk:
                    data["dividendYield"] = _parse_num(v)
                elif "beta" in lk:
                    data["beta"] = _parse_num(v)
                elif "52" in lk and "high" in lk:
                    data["high52"] = _parse_num(v)
                elif "52" in lk and "low" in lk:
                    data["low52"] = _parse_num(v)
                elif "sector" in lk:
                    data["sector"] = v

            if not _has_core_metrics(data):
                return None

            if any(val is not None for val in data.values()):
                sync_data_to_supabase(ticker, data)
                return _ret(data, {"fetchedAt": now_ts, "source": "mubasher", "servedFrom": "live_mubasher"})
        except Exception:
            return None

        return None

    def _try_eodhd() -> Any:
        api_key = os.getenv("EODHD_API_KEY")
        if not api_key:
            return None
        try:
            url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_key}&fmt=json"
            with urllib.request.urlopen(url, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw)

            general = payload.get("General") if isinstance(payload, dict) else None
            highlights = payload.get("Highlights") if isinstance(payload, dict) else None

            if isinstance(general, dict) or isinstance(highlights, dict):
                data = {
                    "marketCap": (highlights or {}).get("MarketCapitalization"),
                    "peRatio": (highlights or {}).get("PERatio"),
                    "eps": (highlights or {}).get("EarningsShare"),
                    "sector": (general or {}).get("Sector"),
                    "beta": (highlights or {}).get("Beta"),
                    "dividendYield": (highlights or {}).get("DividendYield"),
                    "high52": (highlights or {}).get("52WeekHigh"),
                    "low52": (highlights or {}).get("52WeekLow"),
                    "name": (general or {}).get("Name"),
                    "currency": (general or {}).get("CurrencyCode"),
                    "country": (general or {}).get("CountryName"),
                }

                if any(v is not None for v in data.values()):
                    sync_data_to_supabase(ticker, data)
                    return _ret(data, {"fetchedAt": now_ts, "source": "eodhd", "servedFrom": "live_eodhd"})
        except Exception:
            return None

        return None

    if src in {"mubasher", "auto"}:
        got = _try_mubasher_egx()
        if got is not None:
            return got

        if src == "mubasher":
            return _ret({}, {"fetchedAt": now_ts, "status": "error", "source": "mubasher", "servedFrom": "error"})

    if src in {"tradingview", "auto"}:
        got = _try_tradingview()
        if got is not None:
            return got

    return _ret({}, {"fetchedAt": now_ts, "status": "error", "source": src, "servedFrom": "error"})


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names expected from EODHD
    # EODHD typically returns: open, high, low, close, adjusted_close, volume
    cols = {c.lower(): c for c in df.columns}

    close_col = cols.get("close")
    open_col = cols.get("open")
    high_col = cols.get("high")
    low_col = cols.get("low")
    volume_col = cols.get("volume")

    if not all([close_col, open_col, high_col, low_col, volume_col]):
        # Fallback if some columns missing? For now require all for candles.
        # But if just Close/Volume exist (minimal), we might fail. 
        # API usually returns all.
        if close_col is None or volume_col is None:
             raise ValueError("Missing required columns: close/volume")

    # Create a clean dataframe with required columns
    out = pd.DataFrame(index=df.index)
    out["Close"] = df[close_col]
    out["Volume"] = df[volume_col]
    
    # Store others if available
    if open_col: out["Open"] = df[open_col]
    if high_col: out["High"] = df[high_col]
    if low_col: out["Low"] = df[low_col]

    out["SMA_50"] = out["Close"].rolling(window=50, min_periods=1).mean()
    out["SMA_200"] = out["Close"].rolling(window=200, min_periods=1).mean()
    
    # EMA
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["EMA_200"] = out["Close"].ewm(span=200, adjust=False).mean()

    # MACD (12, 26, 9)
    ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_12 - ema_26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    bb_sma20 = out["Close"].rolling(window=20).mean()
    bb_std20 = out["Close"].rolling(window=20).std()
    out["BB_Upper"] = bb_sma20 + (2 * bb_std20)
    out["BB_Lower"] = bb_sma20 - (2 * bb_std20)

    delta = out["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()

    # Avoid division by zero
    rs = gain / loss.replace(0.0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    # pct_change(fill_method=None) to avoid FutureWarning, then fillna(0)
    out["Momentum"] = out["Close"].pct_change(fill_method=None).fillna(0)

    out["VOL_SMA20"] = out["Volume"].rolling(window=20, min_periods=1).mean()
    out["R_VOL"] = (out["Volume"] / out["VOL_SMA20"].replace(0.0, np.nan)).fillna(0.0)

    if "High" in out.columns and "Low" in out.columns:
        high = out["High"].astype(float)
        low = out["Low"].astype(float)
        close = out["Close"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
        out["ATR_14"] = atr

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_sm = tr.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
        plus_dm_sm = pd.Series(plus_dm, index=out.index).ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
        minus_dm_sm = pd.Series(minus_dm, index=out.index).ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()

        plus_di = 100 * (plus_dm_sm / tr_sm.replace(0.0, np.nan))
        minus_di = 100 * (minus_dm_sm / tr_sm.replace(0.0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
        out["ADX_14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean().fillna(0.0)

        lowest_low = low.rolling(window=14, min_periods=1).min()
        highest_high = high.rolling(window=14, min_periods=1).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0.0, np.nan)
        out["STOCH_K"] = stoch_k.fillna(0.0)
        out["STOCH_D"] = out["STOCH_K"].rolling(window=3, min_periods=1).mean()

        tp = (high + low + close) / 3
        tp_sma = tp.rolling(window=20, min_periods=1).mean()
        mean_dev = (tp - tp_sma).abs().rolling(window=20, min_periods=1).mean()
        out["CCI_20"] = ((tp - tp_sma) / (0.015 * mean_dev.replace(0.0, np.nan))).fillna(0.0)

        pv = tp * out["Volume"].astype(float)
        vol_sum = out["Volume"].astype(float).rolling(window=20, min_periods=1).sum()
        out["VWAP_20"] = pv.rolling(window=20, min_periods=1).sum() / vol_sum.replace(0.0, np.nan)
        out["VWAP_20"] = out["VWAP_20"].fillna(0.0)
    else:
        out["ATR_14"] = 0.0
        out["ADX_14"] = 0.0
        out["STOCH_K"] = 0.0
        out["STOCH_D"] = 0.0
        out["CCI_20"] = 0.0
        out["VWAP_20"] = 0.0

    out["ROC_12"] = out["Close"].pct_change(periods=12, fill_method=None).mul(100).fillna(0.0)

    # Bollinger-derived features (percent B and relative width)
    if "BB_Upper" in out.columns and "BB_Lower" in out.columns:
        width = out["BB_Upper"] - out["BB_Lower"]
        out["BB_PctB"] = (
            (out["Close"] - out["BB_Lower"]) / width.replace(0.0, np.nan)
        ).fillna(0.0)
        out["BB_Width"] = (
            width / out["Close"].replace(0.0, np.nan)
        ).fillna(0.0)
    else:
        out["BB_PctB"] = 0.0
        out["BB_Width"] = 0.0

    # On-Balance Volume and simple slope
    price_delta = out["Close"].diff()
    direction = np.sign(price_delta).fillna(0.0)
    obv = (direction * out["Volume"]).cumsum()
    out["OBV"] = obv.fillna(0.0)
    out["OBV_Slope"] = out["OBV"].diff().fillna(0.0)

    # Distance from rolling high/low (context window)
    rolling_high = out["Close"].rolling(window=100, min_periods=1).max()
    rolling_low = out["Close"].rolling(window=100, min_periods=1).min()
    out["Dist_From_High"] = (
        (out["Close"] / rolling_high.replace(0.0, np.nan)) - 1.0
    ).fillna(0.0)
    out["Dist_From_Low"] = (
        (out["Close"] / rolling_low.replace(0.0, np.nan)) - 1.0
    ).fillna(0.0)

    # Z-score of price vs rolling mean/std
    rolling_mean = out["Close"].rolling(window=50, min_periods=1).mean()
    rolling_std = out["Close"].rolling(window=50, min_periods=1).std()
    out["Z_Score"] = (
        (out["Close"] - rolling_mean) / rolling_std.replace(0.0, np.nan)
    ).fillna(0.0)

    # Candle geometry: body and shadows (if OHLC are available)
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

    # Calendar/time features
    if isinstance(out.index, pd.DatetimeIndex):
        out["Day_Of_Week"] = out.index.dayofweek.astype(int)
        out["Day_Of_Month"] = out.index.day.astype(int)
    else:
        out["Day_Of_Week"] = 0
        out["Day_Of_Month"] = 0

    # Lagged values and differences (memory features)
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

    # Do not dropna here, let prepare_for_ai handle it so we can drop columns if needed
    return out

def add_trade_levels(df: pd.DataFrame, risk_reward_ratio: float = 2.0) -> Tuple[float, float]:
    """
    Calculates Take Profit (T.P) and Stop Loss (S.L) based on ATR (Volatility).
    """
    if df.empty or "ATR_14" not in df.columns:
        return 0.0, 0.0
    
    current_price = float(df["Close"].iloc[-1])
    current_atr = float(df["ATR_14"].iloc[-1])
    
    if not np.isfinite(current_atr) or current_atr <= 0:
        # Fallback to simple percentage if ATR is zero or invalid
        return round(current_price * 1.05, 2), round(current_price * 0.97, 2)

    # Stop Loss (S.L) = Current Price - (2.0 * ATR)
    stop_loss = current_price - (2.0 * current_atr)
    
    # Ensure stop_loss is positive and typically at least 1-2% below current price
    if stop_loss > current_price * 0.99: 
        stop_loss = current_price * 0.98
    elif stop_loss < current_price * 0.5: # Extreme ATR check
        stop_loss = current_price * 0.95 
    
    # Target Price (T.P) = Current Price + (2.0 * ATR * risk_reward_ratio)
    # Keeping ATR context for target but user specifically mentioned Stop Loss.
    target_price = current_price + (2.0 * current_atr * risk_reward_ratio)
    
    # Cap target price at reasonable levels
    max_target = current_price * 1.30  # Max 30% profit target
    min_target = current_price * 1.03  # Min 3% profit target
    
    if target_price > max_target:
        target_price = max_target
    elif target_price < min_target:
        target_price = min_target

    return round(float(target_price), 2), round(float(max(stop_loss, 0.01)), 2)


def prepare_for_ai(df: pd.DataFrame, target_pct: float = 0.10, stop_loss_pct: float = 0.05, look_forward_days: int = 20, drop_labels: bool = True) -> pd.DataFrame:
    """
    Sniper Strategy Labeling with configurable parameters.
    Default target lowered to 10% for better sample distribution.
    """
    if df.empty: return df
    out = df.copy()
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=look_forward_days)
    
    # Use lowercase columns if needed or standardizing to capitalized
    close_col = "Close" if "Close" in out.columns else "close"
    open_col = "Open" if "Open" in out.columns else "open"
    high_col = "High" if "High" in out.columns else "high"
    low_col = "Low" if "Low" in out.columns else "low"
    
    # Fix Gap Trap: Entry is Next Day Open
    out['next_open'] = out[open_col].shift(-1)
    
    # Future High/Low: Start checking from next day (shift -1)
    # This aligns the window [t+1, t+1+look_forward] to row t
    out['future_high'] = out[high_col].shift(-1).rolling(window=indexer).max()
    out['future_low'] = out[low_col].shift(-1).rolling(window=indexer).min()
    
    out['Target'] = 0
    
    # Calculate Target/Stop based on Entry Price (Next Open), not Current Close
    hit_target = out['future_high'] >= (out['next_open'] * (1 + target_pct))
    safe_from_stop = out['future_low'] > (out['next_open'] * (1 - stop_loss_pct))
    
    out.loc[hit_target & safe_from_stop, 'Target'] = 1
    
    # Cleanup and dropna
    out.drop(columns=['future_high', 'future_low', 'next_open'], inplace=True, errors='ignore')
    
    # Keep more rows by filling numeric gaps
    # Force common fundamental columns to numeric if they exist to prevent LightGBM dtype errors
    for c in ["marketCap", "peRatio", "eps", "dividendYield"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')

    # Handle categorical fundamental indicators if present
    for c in ["sector", "industry"]:
        if c in out.columns:
            # Force to string then category to handle varied input types gracefully
            out[c] = out[c].fillna("Unknown").astype(str).astype("category")

    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    
    # Drop rows where we couldn't see the future only if requested
    if drop_labels:
        out = out.iloc[:-look_forward_days].copy()
    
    return out


@dataclass
class PredictionResult:
    precision: float
    tomorrow_prediction: int
    last_close: float
    last_date: str
    predictions: List[Dict[str, Any]]


def _clamp_int(v: Any, *, min_v: int, max_v: int) -> int:
    try:
        iv = int(v)
    except Exception:
        raise ValueError("Invalid int")
    if iv < min_v:
        return min_v
    if iv > max_v:
        return max_v
    return iv


def _ensure_feature_columns(df: pd.DataFrame, features: List[str]) -> None:
    missing = [f for f in features if f not in df.columns]
    if not missing:
        return
    
    # Suppress fragmentation warning – it's harmless here
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        
        # Case-insensitive recovery: if model expects 'close' but we have 'Close', copy it!
        existing_lower = {c.lower(): c for c in df.columns}
        
        for name in missing:
            # 1. Try case-insensitive match
            lower_name = name.lower()
            if lower_name in existing_lower:
                df[name] = df[existing_lower[lower_name]]
            # 2. Fallback to zero or "Unknown" for categories
            if name in ["sector", "industry"]:
                df[name] = "Unknown"
                df[name] = df[name].astype("category")
            else:
                df[name] = 0.0


def _clamp_float(v: Any, *, min_v: float, max_v: float) -> float:
    try:
        fv = float(v)
    except Exception:
        raise ValueError("Invalid float")
    if fv < min_v:
        return min_v
    if fv > max_v:
        return max_v
    return fv


def _sanitize_rf_params(
    raw: Optional[Dict[str, Any]],
    *,
    preset: Optional[str],
    train_len: int,
    default_min_split: int,
) -> Dict[str, Any]:
    base: Dict[str, Any]
    if (preset or "").lower() == "fast":
        base = {
            "n_estimators": 80,
            "max_depth": 10,
            "min_samples_split": max(2, min(200, max(10, int(train_len * 0.25)))),
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": -1,
            "random_state": 1,
        }
    elif (preset or "").lower() == "accurate":
        base = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": max(2, default_min_split),
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": -1,
            "random_state": 1,
        }
    else:
        base = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": max(2, default_min_split),
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": -1,
            "random_state": 1,
        }

    if not raw:
        return base

    out = dict(base)

    for k, v in raw.items():
        try:
            if k in ("n_estimators", "verbose"):
                out[k] = _clamp_int(v, min_v=1, max_v=2000)
            elif k in ("random_state", "max_leaf_nodes", "n_jobs"):
                if v is None:
                    out[k] = None
                else:
                    out[k] = _clamp_int(v, min_v=-1 if k == "n_jobs" else 0, max_v=10_000)
            elif k == "max_depth":
                if v is None:
                    out[k] = None
                else:
                    out[k] = _clamp_int(v, min_v=1, max_v=256)
            elif k in ("min_samples_split", "min_samples_leaf"):
                if isinstance(v, float) or isinstance(v, int):
                    if 0 < float(v) < 1:
                        out[k] = _clamp_float(v, min_v=0.0001, max_v=0.9)
                    else:
                        out[k] = _clamp_int(v, min_v=2 if k == "min_samples_split" else 1, max_v=10_000)
            elif k in ("min_weight_fraction_leaf", "min_impurity_decrease", "ccp_alpha"):
                out[k] = _clamp_float(v, min_v=0.0, max_v=10.0)
            elif k == "max_features":
                if v is None:
                    out[k] = None
                elif isinstance(v, (int, float)):
                    if 0 < float(v) <= 1:
                        out[k] = _clamp_float(v, min_v=0.01, max_v=1.0)
                    else:
                        out[k] = _clamp_int(v, min_v=1, max_v=256)
                elif isinstance(v, str):
                    vv = v.strip().lower()
                    if vv in ("sqrt", "log2"):
                        out[k] = vv
                    elif vv == "none":
                        out[k] = None
            elif k in ("bootstrap", "oob_score", "warm_start"):
                out[k] = bool(v)
            elif k == "criterion":
                if isinstance(v, str):
                    vv = v.strip().lower()
                    if vv in ("gini", "entropy", "log_loss"):
                        out[k] = vv
            elif k == "class_weight":
                if v is None or isinstance(v, (str, dict)):
                    out[k] = v
            elif k == "max_samples":
                if v is None:
                    out[k] = None
                elif isinstance(v, (int, float)):
                    if 0 < float(v) <= 1:
                        out[k] = _clamp_float(v, min_v=0.01, max_v=1.0)
                    else:
                        out[k] = _clamp_int(v, min_v=1, max_v=10_000_000)
        except Exception:
            continue

    return out


def train_and_predict(
    df: pd.DataFrame,
    rf_params: Optional[Dict[str, Any]] = None,
    rf_preset: Optional[str] = None,
    exchange: Optional[str] = None,
    model_name: Optional[str] = None,
    target_pct: float = 0.15,
    stop_loss_pct: float = 0.05,
    look_forward_days: int = 20,
    buy_threshold: float = 0.40,
) -> Tuple[Any, List[str], pd.DataFrame, pd.Series, float, float]:
    """
    Core prediction logic. 
    1. Tries to load a pre-trained model for the exchange.
    2. If not found, trains a fresh RandomForest model on the fly.
    """
    preset = (rf_preset or "").lower()
    
    # Path for pre-trained model. Prefer explicit model_name if provided,
    # otherwise fall back to model_{exchange}.pkl convention.
    api_dir = os.path.dirname(os.path.abspath(__file__))
    if model_name:
        model_path = os.path.join(api_dir, "models", model_name)
    else:
        model_filename = f"model_{exchange}.pkl" if exchange else None
        model_path = os.path.join(api_dir, "models", model_filename) if model_filename else None
    
    loaded_model = None
    predictors = RF_PREDICTORS
    lgbm_artifact_loaded = False
    
    if model_path and os.path.exists(model_path):
        # Try cached model first
        cached = _get_model_cached(model_path)
        if cached:
            loaded_model, cached_predictors, cached_is_lgbm = cached
            if cached_predictors:
                predictors = cached_predictors
            lgbm_artifact_loaded = bool(cached_is_lgbm)
            print(f"Using cached model {model_path} with {len(predictors) if predictors else 0} predictors")
        else:
            try:
                with open(model_path, "rb") as f:
                    loaded_model = pickle.load(f)

                # New lightweight model artifact format from train_exchange_model.py
                lgbm_artifact_loaded = False
                if isinstance(loaded_model, dict) and loaded_model.get("kind") == "lgbm_booster":
                    if lgb is None:
                        raise ValueError("lightgbm is required to load lgbm_booster artifacts")

                    artifact = loaded_model
                    model_str = artifact.get("model_str")
                    if not isinstance(model_str, str) or not model_str.strip():
                        raise ValueError("Invalid lgbm_booster artifact: missing model_str")

                    booster = lgb.Booster(model_str=model_str)
                    threshold = artifact.get("threshold")
                    loaded_model = _LgbmBoosterClassifier(booster, threshold if isinstance(threshold, (int, float)) else 0.5)
                    lgbm_artifact_loaded = True

                    f_names = artifact.get("feature_names")
                    if isinstance(f_names, list) and f_names:
                        predictors = f_names
                    else:
                        try:
                            predictors = list(booster.feature_name())
                        except Exception:
                            predictors = LGBM_PREDICTORS

                    if not predictors:
                        raise ValueError(
                            "No overlapping predictors between trained LightGBM booster and current feature set. Please retrain the model."
                        )

                    print(f"Loaded LightGBM booster artifact from {model_path} with {len(predictors)} predictors")
                    
                    # Extract target settings from artifact
                    target_pct = artifact.get("target_pct", target_pct)
                    stop_loss_pct = artifact.get("stop_loss_pct", stop_loss_pct)
                    look_forward_days = artifact.get("look_forward_days", look_forward_days)

                    _ensure_feature_columns(df, predictors)

                # Identify predictors based on model type
                if (not lgbm_artifact_loaded) and LGBMClassifier and isinstance(loaded_model, LGBMClassifier):
                    model_features = None
                    # Prefer the feature names stored with the trained model to avoid
                    # shape mismatches when the global feature list changes.
                    try:
                        model_features = getattr(loaded_model, "feature_name_", None)
                    except Exception:
                        model_features = None

                    if (not model_features) and hasattr(loaded_model, "booster_"):
                        try:
                            booster = getattr(loaded_model, "booster_", None)
                            if booster is not None:
                                model_features = booster.feature_name()
                        except Exception:
                            model_features = None

                    if model_features:
                        predictors = list(model_features)
                    else:
                        predictors = LGBM_PREDICTORS

                    if not predictors:
                        raise ValueError(
                            "No overlapping predictors between trained LightGBM model and current feature set. Please retrain the model."
                        )

                    print(f"Loaded pre-trained LightGBM model from {model_path} with {len(predictors)} predictors")
                    _ensure_feature_columns(df, predictors)
                elif not lgbm_artifact_loaded:
                    predictors = [p for p in RF_PREDICTORS if p in df.columns]
                    print(f"Loaded pre-trained model for {exchange}")

                # Save to cache
                _set_model_cache(model_path, loaded_model, predictors, lgbm_artifact_loaded)
            except Exception as e:
                print(f"Failed to load pre-trained model {model_path}: {e}")
                loaded_model = None

    if loaded_model and predictors:
        _ensure_feature_columns(df, predictors)

    if preset == "fast" and len(df) > 600:
        df = df.iloc[-600:].copy()

    # If we have a loaded model, we don't need to "train" but we still need to generate 
    # test predictions and precision for the UI.
    if loaded_model:
        if len(df) < 100: # Smaller threshold for pre-trained validation
            # If really short, just return the last prediction with dummy precision
            last_row = df.iloc[[-1]][predictors]
            pred = int(loaded_model.predict(last_row)[0])
            return loaded_model, predictors, df.iloc[[-1]], pd.Series([pred], index=[df.index[-1]]), 0.8
            
        # For precision metrics, we MUST exclude the trailing look_forward_days rows 
        # because we don't know the ground truth for them yet.
        test_data_pool = df.iloc[:-look_forward_days].copy()

        # Relax test size for pre-trained validation
        # Use all available data for testing to satisfy the "test all rows" request
        test_size = len(test_data_pool)
        test = test_data_pool.iloc[-test_size:]
        
        if hasattr(loaded_model, "predict_proba"):
            try:
                probs = loaded_model.predict_proba(test[predictors])[:, 1]
                preds = (probs >= buy_threshold).astype(int)
            except Exception as e:
                # Catch categorical mismatch specifically for LightGBM
                if "categorical_feature do not match" in str(e):
                    print(f"DEBUG: Categorical mismatch for model. Falling back to simple predict.")
                    try:
                        # Try simple predict as fallback
                        preds = loaded_model.predict(test[predictors])
                    except:
                        # Last resort: random-like or neutral prediction if model is totally broken for this stock
                        preds = np.zeros(len(test), dtype=int)
                else:
                    try:
                        preds = loaded_model.predict(test[predictors])
                    except:
                        preds = np.zeros(len(test), dtype=int)
        else:
            try:
                preds = loaded_model.predict(test[predictors])
            except:
                preds = np.zeros(len(test), dtype=int)

        preds = pd.Series(preds, index=test.index)
        precision = precision_score(test["Target"], preds, zero_division=0)
        
        # Calculate Earn Percentage (Average Return Per Trade)
        buy_indices = preds[preds == 1].index
        earn_pct = 0.0
        if len(buy_indices) > 0:
            wins = (test.loc[buy_indices, "Target"] == 1).sum()
            losses = len(buy_indices) - wins
            win_rate = wins / len(buy_indices)
            
            # Average return per trade = (win_rate × target_gain) - (loss_rate × stop_loss)
            earn_pct = (win_rate * target_pct) - ((1 - win_rate) * stop_loss_pct)
        
        return loaded_model, predictors, test, preds, float(precision), float(earn_pct * 100)

    # If a specific model was requested but not loaded, stop here to avoid on-the-fly training
    if model_name:
        raise ValueError(
            f"Requested model '{model_name}' not loaded. Please place the .pkl file and retry."
        )

    # Fallback to existing training logic (only when no explicit model_name)
    use_lgbm = (LGBMClassifier is not None)
    
    if use_lgbm:
        # Use rich feature set for Gradient Boosting
        predictors = [p for p in LGBM_PREDICTORS if p in df.columns]
    else:
        # Fallback to smaller set for RF
        predictors = [p for p in RF_PREDICTORS if p in df.columns]

    if len(df) < 120:
        raise ValueError("Not enough data to train. Need at least ~120 rows.")

    if preset == "fast":
        test_size = 60 if len(df) > 260 else max(40, int(len(df) * 0.2))
    else:
        test_size = 100 if len(df) > 400 else max(40, int(len(df) * 0.2))
        
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]

    if use_lgbm:
        # Validation split for early stopping logic
        val_size = max(20, int(len(train) * 0.15))
        train_inner = train.iloc[:-val_size]
        val_inner = train.iloc[-val_size:]
        
        # Configure LightGBM with robust parameters
        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            importance_type='gain',
            verbose=-1
        )
        
        callbacks = []
        if lgb:
            # Use LightGBM callbacks for early stopping (requires validation set)
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))
            
        try:
            model.fit(
                train_inner[predictors], train_inner["Target"],
                eval_set=[(val_inner[predictors], val_inner["Target"])],
                eval_metric="logloss",
                callbacks=callbacks
            )
            print(f"Trained LGBM (Early Stopping @ {model.best_iteration_ if hasattr(model, 'best_iteration_') else 'N/A'})")
        except Exception as e:
            print(f"LGBM Training failed ({e}), falling back to RF...")
            use_lgbm = False # Trigger fallback below
            
    if not use_lgbm:
        # Legacy Random Forest fallback
        min_split = min(100, max(2, int(len(train) * 0.2)))
    
        model_kwargs = _sanitize_rf_params(
            rf_params,
            preset=rf_preset,
            train_len=len(train),
            default_min_split=min_split,
        )
    
        model = RandomForestClassifier(**model_kwargs)
        model.fit(train[predictors], train["Target"])

    # Use probability threshold instead of hard default 0.5
    # To make it bolder, we use the passed buy_threshold
    try:
        probs = model.predict_proba(test[predictors])[:, 1]
        preds = (probs >= buy_threshold).astype(int)
    except Exception:
        # Fallback if model doesn't support predict_proba
        preds = model.predict(test[predictors])
        
    preds = pd.Series(preds, index=test.index)
    precision = precision_score(test["Target"], preds, zero_division=0)

    # Calculate Earn Percentage (Average Return Per Trade)
    buy_indices = preds[preds == 1].index
    earn_pct = 0.0
    if len(buy_indices) > 0:
        wins = (test.loc[buy_indices, "Target"] == 1).sum()
        losses = len(buy_indices) - wins
        win_rate = wins / len(buy_indices)
        
        # Average return per trade = (win_rate × target_gain) - (loss_rate × stop_loss)
        earn_pct = (win_rate * target_pct) - ((1 - win_rate) * stop_loss_pct)

    # --- ADAPTIVE LEARNING LOGGING ---
    # Log the LATEST prediction (today's signal) for future retraining
    try:
        if loaded_model:  # Only log if using a stable/loaded model (not on-the-fly random forest)
            from api.adaptive_learning import ActiveLearner
            
            # Get latest row
            last_idx = test.index[-1]
            last_row = test.iloc[-1]
            last_pred = int(preds.iloc[-1])
            
            # Get probability if available
            last_prob = 0.5
            if 'probs' in locals():
                last_prob = float(probs[-1])
            elif hasattr(preds, 'iloc'):
                last_prob = float(preds.iloc[-1]) # Binary confidence if prob unavailable

            # Extract features used
            # We need to pass the raw feature values used by the model
            # Note: 'train_and_predict' predictors list matches columns in 'test'
            feat_values = last_row[predictors].values
            
            # We assume the symbol comes from context or df metadata? 
            # 'df' usually doesn't have symbol column if it's the index or dropped.
            # But 'run_pipeline' calls this. 'train_and_predict' doesn't know the symbol explicitly unless passed?
            # 'train_and_predict' doesn't take 'symbol' arg.
            # However, 'run_pipeline' knows it. 
            # We should probably do this logging in 'run_pipeline' instead to have the symbol context.
            # BUT 'run_pipeline' receives the preds. 
            # So I will NOT do it here. I will do it in run_pipeline.
            pass
    except Exception as e:
        # print(f"Adaptive logging warning: {e}")
        pass

    return model, predictors, test, preds, float(precision), float(earn_pct * 100)


def run_pipeline(
    api_key: str,
    ticker: str,
    from_date: str = "2020-01-01",
    include_fundamentals: bool = True,
    tolerance_days: int = 0,
    exchange: Optional[str] = None,
    force_local: bool = False,
    rf_params: Optional[Dict[str, Any]] = None,
    rf_preset: Optional[str] = None,
    model_name: Optional[str] = None,
    target_pct: float = 0.15,
    stop_loss_pct: float = 0.05,
    look_forward_days: int = 20,
    buy_threshold: float = 0.40,
) -> Dict[str, Any]:
    api = APIClient(api_key)

    last_error: Optional[Exception] = None
    prices_ai: Optional[pd.DataFrame] = None
    selected_symbol: Optional[str] = None
    last_candidate_len: Optional[int] = None
    min_required = 60 if model_name else 120

    for sym in _candidate_symbols(ticker, exchange):
        try:
            raw = get_stock_data_eodhd(
                api, 
                sym, 
                from_date=from_date, 
                tolerance_days=tolerance_days, 
                exchange=exchange,
                force_local=force_local
            )
            if raw is None or raw.empty:
                raise ValueError(
                    f"No historical data available for {sym} from local cache. "
                    "Try sync first or disable force_local."
                )

            # Trim early to reduce heavy feature computation during scanning
            if force_local and len(raw) > 500:
                raw = raw.iloc[-500:].copy()
            # Generate a rich feature set to match models trained by train_exchange_model.py.
            # This is critical for "max" models that rely on TA/massive features.
            # USE CACHED VERSION AS IT IS VERY EXPENSIVE
            feat = _get_indicator_cache().get_with_massive_features(sym, exchange or "US", raw)
            
            # Re-ensure basic indicators are there (without wiping massive features)
            try:
                # Only call if we don't have basic indicators like SMA_50 yet
                if "SMA_50" not in feat.columns:
                    feat_with_ind = _get_data_with_indicators_cached(sym, exchange or "US", feat, add_technical_indicators)
                    if feat_with_ind is not None and not feat_with_ind.empty:
                        # Merge instead of overwrite to keep massive features
                        for col in feat_with_ind.columns:
                            if col not in feat.columns:
                                feat[col] = feat_with_ind[col]
            except Exception:
                pass

            # NEW: Add Market Context features (Level 4)
            # Infer exchange from symbol if not provided
            current_exchange = exchange or (sym.split('.')[-1] if '.' in sym else "US")
            index_sym = "EGX30.INDX" if current_exchange in ["EGX", "CA"] else "GSPC.INDX"
            
            # Fetch index data (simple internal cache / optimized fetch could be added here)
            # For now, quick fetch
            market_df = None
            try:
                if supabase:
                    m_res = supabase.table("stock_prices").select("date, close").eq("symbol", index_sym).order("date", desc=False).execute()
                    if m_res.data:
                        market_df = pd.DataFrame(m_res.data)
                        market_df["date"] = pd.to_datetime(market_df["date"])
                        market_df = market_df.set_index("date").sort_index()
                        market_df['atr'] = market_df['close'].pct_change().rolling(20).std().fillna(0)
            except Exception:
                pass # Silently fail back to defaults if index not found
            
            feat = add_market_context(feat, market_df)

            # --- PHASE EXTRA: Incorporate Fundamentals ---
            if include_fundamentals:
                try:
                    funds = get_company_fundamentals(sym) or {}
                    if funds:
                        feat["marketCap"] = _finite_float(funds.get("marketCap"))
                        feat["peRatio"] = _finite_float(funds.get("peRatio"))
                        feat["eps"] = _finite_float(funds.get("eps"))
                        feat["dividendYield"] = _finite_float(funds.get("dividendYield"))
                        feat["sector"] = funds.get("sector")
                        feat["industry"] = funds.get("industry")
                        # Handle categorical explicitly for LGBM
                        for c in ["sector", "industry"]:
                            if c in feat.columns:
                                feat[c] = feat[c].fillna("Unknown").astype("category")
                except Exception as e:
                    print(f"Warning: Fundamental merge failed for {sym}: {e}")

            # 1. Try with all features, keeping trailing rows for prediction
            candidate = prepare_for_ai(feat, target_pct=target_pct, stop_loss_pct=stop_loss_pct, look_forward_days=look_forward_days, drop_labels=False)
            
            # 2. If not enough data (likely due to SMA_200 NaNs on a short history), 
            #    drop SMA_200 and try again.
            if len(candidate) < 120 and "SMA_200" in feat.columns:
                print(f"Warning: Insufficient data for SMA_200 (rows={len(candidate)}). Retrying without it.")
                feat_reduced = feat.drop(columns=["SMA_200"])
                candidate = prepare_for_ai(feat_reduced, target_pct=target_pct, stop_loss_pct=stop_loss_pct, look_forward_days=look_forward_days)

            last_candidate_len = len(candidate)

            if len(candidate) >= min_required:
                prices_ai = candidate
                selected_symbol = sym
                break
            
            last_error = ValueError(f"Not enough rows after feature engineering. Got {len(candidate)}")
        except Exception as e:
            last_error = e

    if prices_ai is None:
        suffix = ""
        if selected_symbol:
            suffix = f" (tried {selected_symbol})"
        if last_candidate_len is not None:
            raise ValueError(
                f"Insufficient historical data for {ticker}{suffix}. Need at least ~{min_required} rows after feature engineering, got {last_candidate_len}."
            ) from last_error
        raise ValueError(
            f"Insufficient historical data or symbol not available on your API plan for {ticker}{suffix}. Try specifying exchange, e.g. AAPL.US"
        ) from last_error

    # For scanning mode (FAST SCAN), limit rows to shrink indicator computation and speed up
    # But if we are running a specific MODEL TEST (model_name is set), we want more history.
    limit_rows = 10000 if model_name else 450
    if force_local and len(prices_ai) > limit_rows:
        prices_ai = prices_ai.iloc[-limit_rows:].copy()

    # For training (on-the-fly), we MUST exclude the trailing look_forward_days rows 
    # because they don't have accurate 'Target' labels yet (future is unknown).
    prices_ai_training = prices_ai.iloc[:-look_forward_days].copy() if not model_name else prices_ai

    model, predictors, test_df, preds, precision, earn_percentage = train_and_predict(
        df=prices_ai_training,
        rf_params=rf_params,
        rf_preset=rf_preset,
        exchange=exchange,
        model_name=model_name,
        target_pct=target_pct,
        stop_loss_pct=stop_loss_pct,
        look_forward_days=look_forward_days,
        buy_threshold=buy_threshold,
    )

    last_row = prices_ai.iloc[[-1]][predictors]
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(last_row)[:, 1][0]
            tomorrow_prediction = 1 if prob >= buy_threshold else 0
        except Exception as e:
            msg = str(e)
            if "categorical_feature do not match" in msg:
                print(f"DEBUG: Categorical mismatch in final prediction for {selected_symbol or ticker}. Falling back.")
            try:
                tomorrow_prediction = int(model.predict(last_row)[0])
            except:
                tomorrow_prediction = 0 # Default to no action if all fails
    else:
        try:
            tomorrow_prediction = int(model.predict(last_row)[0])
        except:
            tomorrow_prediction = 0

    # --- Phase 7: Finalize Data ---
    last_close = float(prices_ai.iloc[-1]["Close"])
    last_date = str(pd.to_datetime(prices_ai.index[-1]).date())

    # --- ADAPTIVE LEARNING LOGGING ---
    # Log the LATEST prediction (today's signal) for future manual retraining/review
    # Only if using a specific loaded model (active production model)
    if model_name: 
        try:
            from api.adaptive_learning import ActiveLearner
            # Use 'EODHD' or 'US' if exchange is not explicitly set
            logging_exchange = exchange or "EGX" 
            learner = ActiveLearner(exchange=logging_exchange)
            
            # Get probability for confidence
            pred_conf = 0.5
            if hasattr(model, "predict_proba"):
                try:
                    p = model.predict_proba(last_row)[:, 1][0]
                    pred_conf = float(p)
                except: 
                    pred_conf = float(tomorrow_prediction)
            else:
                 pred_conf = float(tomorrow_prediction)

            # Extract features (values only)
            # last_row is a DataFrame with 1 row
            feat_vals = last_row.values[0]

            # Prepare data for scan_results
            data = {
                "user_id": None, # This is a server-side log, no user_id by default
                "symbol": selected_symbol or ticker,
                "exchange": logging_exchange,
                "name": selected_symbol or ticker,
                "model_name": model_name,
                "country": "Egypt", # Default country for this app context
                "last_close": last_close,
                "precision": pred_conf,
                "signal": "BUY" if tomorrow_prediction == 1 else "SELL",
                "status": "open",
                "entry_price": last_close,
                "features": json.dumps(feat_vals.tolist()) if hasattr(feat_vals, "tolist") else json.dumps(list(feat_vals)),
                "created_at": datetime.now().isoformat()
            }
            supabase.table("scan_results").insert(data).execute()
        except Exception as e:
            print(f"Adaptive logging warning: {e}")

    # --- Phase 8: Explainable AI (Top 3 Reasons) ---
    top_reasons = get_top_reasons(model, predictors)

    # Fill NaNs for indicators to avoid JSON serialization issues or frontend gaps
    # Forward fill then backward fill to cover edges
    cols_to_fill = ["EMA_50", "EMA_200", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "RSI", "Momentum"]
    for c in cols_to_fill:
        if c in prices_ai.columns:
            prices_ai[c] = prices_ai[c].ffill().bfill().fillna(0.0)

    # Create display window
    # User requested "all data", so we return the full dataset provided by the logic
    display_df = prices_ai

    out_preds: List[Dict[str, Any]] = []
    for idx, row in display_df.iterrows():
        # Check if this row has a prediction (was in test set)
        p = preds.get(idx) 
        
        # If p is None (training data), default to 0 (Hold) or similar?
        # User implies they want to see the chart context. 
        # We won't show "Buy" markers for training data to avoiding "perfect fit" confusion.
        prediction_val = int(p) if p is not None else 0
        
        target_val = int(row["Target"]) # Target is future > current

        close_v = _finite_float(row.get("Close"))
        if close_v is None:
            continue

        row_data = {
            "date": str(pd.to_datetime(idx).date()),
            "close": close_v,
            "pred": prediction_val,
            "target": target_val,
            "is_test": p is not None,  # Identify if this row belongs to the test set
        }
        
        # Add OHLC
        if "Open" in row:
            v = _finite_float(row["Open"])
            if v is not None:
                row_data["open"] = v
        if "High" in row:
            v = _finite_float(row["High"])
            if v is not None:
                row_data["high"] = v
        if "Low" in row:
            v = _finite_float(row["Low"])
            if v is not None:
                row_data["low"] = v
        if "Volume" in row:
            v = _finite_float(row["Volume"])
            if v is not None:
                row_data["volume"] = v

        # Add indicators
        if "SMA_50" in row:
            v = _finite_float(row["SMA_50"])
            if v is not None:
                row_data["sma50"] = v
        if "SMA_200" in row:
            v = _finite_float(row["SMA_200"])
            if v is not None:
                row_data["sma200"] = v
        
        if "EMA_50" in row:
            v = _finite_float(row["EMA_50"])
            if v is not None:
                row_data["ema50"] = v
        
        # EMA_200 should now be present for almost all rows
        if "EMA_200" in row:
            v = _finite_float(row["EMA_200"])
            if v is not None:
                row_data["ema200"] = v
        
        if "MACD" in row:
            v = _finite_float(row["MACD"])
            if v is not None:
                row_data["macd"] = v
        if "MACD_Signal" in row:
            v = _finite_float(row["MACD_Signal"])
            if v is not None:
                row_data["macd_signal"] = v
        if "BB_Upper" in row:
            v = _finite_float(row["BB_Upper"])
            if v is not None:
                row_data["bb_upper"] = v
        if "BB_Lower" in row:
            v = _finite_float(row["BB_Lower"])
            if v is not None:
                row_data["bb_lower"] = v

        if "RSI" in row:
            v = _finite_float(row["RSI"])
            if v is not None:
                row_data["rsi"] = v
        if "Momentum" in row:
            v = _finite_float(row["Momentum"])
            if v is not None:
                row_data["momentum"] = v
        
        out_preds.append(row_data)

    fundamentals = get_company_fundamentals(selected_symbol or ticker) if include_fundamentals else {}
    fundamentals = _sanitize_fundamentals(fundamentals) if include_fundamentals else {}
    fundamentals_error = None
    if include_fundamentals and not fundamentals:
        fundamentals_error = "Fundamentals not available from yfinance."

    return {
        "ticker": selected_symbol or ticker,
        "precision": precision,
        "earnPercentage": earn_percentage,
        "tomorrowPrediction": tomorrow_prediction,
        "lastClose": last_close,
        "lastDate": last_date,
        "fundamentals": fundamentals,
        "fundamentalsError": fundamentals_error,
        "testPredictions": out_preds,
        "note": "Precision is the % of correct UP predictions in the test set. Higher is better.",
        "topReasons": top_reasons,
    }

# Alias for compatibility
get_stock_data = get_stock_data_eodhd

def batch_check_local_cache(symbol_exchange_list: List[Tuple[str, Optional[str]]]) -> Dict[Tuple[str, Optional[str]], bool]:
    """Efficiently checks multiple symbols in Supabase in a single batch (via cached_tickers)."""
    cached_set = get_cached_tickers()
    results = {}
    for s, e in symbol_exchange_list:
        results[(s, e)] = is_ticker_synced(s, e)
    return results

def batch_get_stock_data(
    api: APIClient,
    ticker_list: List[str],
    from_date: str,
    exchange_list: Optional[List[Optional[str]]] = None,
    force_local: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetches stock data for multiple tickers. 
    Currently sequential but optimized for cloud-first lookups.
    """
    results = {}
    for i, ticker in enumerate(ticker_list):
        ex = exchange_list[i] if exchange_list else None
        try:
            df = get_stock_data_eodhd(api, ticker, from_date, exchange=ex, force_local=force_local)
            if not df.empty:
                results[ticker] = df
        except Exception:
            continue
    return results
