import datetime as dt
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf
import json
from eodhd import APIClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from supabase import create_client, Client

supabase: Optional[Client] = None

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
DEFAULT_CACHE_DIR = os.path.join(BASE_DIR, "api", "data_cache")
DEFAULT_SYMBOLS_DIR = os.path.join(BASE_DIR, "symbols_data")


def _resolve_cache_dir(cache_dir: str) -> str:
    p = (cache_dir or "").strip()
    if not p:
        return DEFAULT_CACHE_DIR
    if os.path.isabs(p):
        return p
    p_norm = p.replace("/", os.sep).replace("\\", os.sep)
    direct = os.path.join(BASE_DIR, p_norm)
    api_joined = os.path.join(BASE_DIR, "api", p_norm)
    if os.path.isdir(direct):
        return direct
    if os.path.isdir(api_joined):
        return api_joined
    return api_joined


def _cache_subdir_for_symbol(symbol: str) -> Optional[str]:
    s = (symbol or "").strip().upper()
    if "." not in s:
        return None
    suffix = s.split(".")[-1]
    if suffix == "CC":
        return "EGX"
    if suffix.isalnum() and 1 <= len(suffix) <= 10:
        return suffix
    return None


def _candidate_cache_paths(cache_dir: str, symbol: str) -> List[str]:
    cache_dir = _resolve_cache_dir(cache_dir)
    key = _safe_cache_key(symbol)
    filename = f"{key}.csv"
    subdir = _cache_subdir_for_symbol(symbol)
    out: List[str] = []
    if subdir:
        out.append(os.path.join(cache_dir, subdir, "prices", filename))
        out.append(os.path.join(cache_dir, subdir, filename))
    else:
        out.append(os.path.join(cache_dir, "prices", filename))
    out.append(os.path.join(cache_dir, filename))
    seen: List[str] = []
    for p in out:
        if p not in seen:
            seen.append(p)
    return seen


def _preferred_cache_path(cache_dir: str, symbol: str) -> str:
    cache_dir = _resolve_cache_dir(cache_dir)
    key = _safe_cache_key(symbol)
    filename = f"{key}.csv"
    subdir = _cache_subdir_for_symbol(symbol)
    if subdir:
        return os.path.join(cache_dir, subdir, "prices", filename)
    return os.path.join(cache_dir, "prices", filename)


def _candidate_fund_cache_paths(cache_dir: str, symbol: str) -> List[str]:
    cache_dir = _resolve_cache_dir(cache_dir)
    filename = f"fund_{_safe_cache_key(symbol)}.json"
    subdir = _cache_subdir_for_symbol(symbol)
    out: List[str] = []
    if subdir:
        out.append(os.path.join(cache_dir, subdir, "fund", filename))
    out.append(os.path.join(cache_dir, "fund", filename))
    seen: List[str] = []
    for p in out:
        if p not in seen:
            seen.append(p)
    return seen


def _preferred_fund_cache_path(cache_dir: str, symbol: str) -> str:
    cache_dir = _resolve_cache_dir(cache_dir)
    filename = f"fund_{_safe_cache_key(symbol)}.json"
    subdir = _cache_subdir_for_symbol(symbol)
    if subdir:
        return os.path.join(cache_dir, subdir, "fund", filename)
    return os.path.join(cache_dir, "fund", filename)


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


def _safe_mkdir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


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
    
    # 1. Split by dot if present
    if "." in t:
        parts = t.split(".")
        s = parts[0]
        e = parts[1]
        # Map known variations
        if e == "CC": e = "EGX"
        if e == "NYSE" or e == "NASDAQ": e = "US"
        return s, e
        
    # 2. Use hint if provided
    if exchange_hint:
        e = exchange_hint.upper()
        if e == "CC": e = "EGX"
        if e == "NYSE" or e == "NASDAQ": e = "US"
        return t, e
        
    # 3. Default fallback (guessing) - mostly US or EGX in this app context
    # If it's 4+ letters and no dots, often US. If 3 letters, could be either.
    # But usually the ticker should have the suffix if it's EGX (like COMI.EGX).
    return t, "US"


def check_local_cache(symbol: str, exchange: Optional[str] = None, cache_dir: str = DEFAULT_CACHE_DIR) -> bool:
    """Checks if data exists in Supabase."""
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

def get_cached_tickers(cache_dir: str = DEFAULT_CACHE_DIR) -> set:
    """Returns a set of all ticker names. Prioritizes Supabase, then scans local cache. Cached for 30s."""
    global _CACHED_TICKERS_SET, _CACHED_TICKERS_TS
    
    import time
    now = time.time()
    
    if _CACHED_TICKERS_SET is not None and (now - _CACHED_TICKERS_TS) < _CACHE_TTL_SECONDS:
        return _CACHED_TICKERS_SET

    print(f"DEBUG: get_cached_tickers computing...")
    found = set()

    # 1. Try Supabase
    _init_supabase()
    if supabase:
        try:
            # Get unique symbol + exchange combinations
            # Note: We use distinct on symbol,exchange if possible, or just symbols.
            # For simplicity, let's get all symbols from fundamentals table as it's smaller.
            res = supabase.table("stock_fundamentals").select("symbol,exchange").execute()
            if res.data:
                for row in res.data:
                    s, e = row['symbol'], row['exchange']
                    found.add(f"{s}.{e}")
        except Exception as e:
            print(f"DEBUG: Supabase ticker fetch failed: {e}")
        
    _CACHED_TICKERS_SET = found
    _CACHED_TICKERS_TS = now
    return found


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
    
    # Fallbacks
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
        df = df.sort_index()

    return df


def get_stock_data_eodhd(
    api: APIClient,
    ticker: str,
    from_date: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    tolerance_days: int = 0,
    exchange: Optional[str] = None,
    force_local: bool = False,
) -> pd.DataFrame:
    cache_dir = _resolve_cache_dir(cache_dir)
    _safe_mkdir(cache_dir)
    
    # Try multiple possible filenames to favor local cache
    possible_names = [ticker]
    if exchange:
        mapping = {"EGX": "EGX", "US": "US", "NYSE": "US", "NASDAQ": "US"}
        suffix = mapping.get(exchange.upper(), exchange.upper())
        possible_names.append(f"{ticker.split('.')[0]}.{suffix}")
        # Legacy/Fallback
        if exchange.upper() == "EGX":
            possible_names.append(f"{ticker.split('.')[0]}.CC")
            possible_names.append(f"{ticker.split('.')[0]}.EGX")

    if "." in ticker:
        base = ticker.split(".")[0]
        # Map back CC -> EGX or US
        if ticker.endswith(".CC"): possible_names.append(f"{base}.EGX")
        possible_names.append(base)
    
    # 0. Try Supabase
    _init_supabase()
    if supabase:
        try:
            s, e = _infer_symbol_exchange(ticker, exchange)
            q = supabase.table("stock_prices").select("date,open,high,low,close,volume").eq("symbol", s).eq("exchange", e)
            
            # Order by date asc
            res = q.order("date", desc=False).execute()
            
            if res.data:
                df = pd.DataFrame(res.data)
                if not df.empty:
                    # Convert to standard format
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()
                    # Filter by from_date
                    df = df[df.index >= pd.to_datetime(from_date)]
                    if not df.empty:
                        print(f"DEBUG: Supabase hit for {s}.{e}")
                        return df
        except Exception as e:
            print(f"Supabase read error for {ticker}: {e}")

    if force_local:
        raise ValueError(f"Symbol {ticker} not found in Supabase.")

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
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_local: bool = False
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.
    """
    # Disk-less: No local file check. 
    # Logic: Always fetch from Yahoo if get_stock_data-Supabase failed.
    if force_local:
        # If force_local means 'only cloud', it should have been handled in get_stock_data.
        # But here we assume we are the 'live' fallback.
        raise ValueError(f"Symbol {ticker} missing in cloud.")

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
            
        df = df.sort_index()
        
        if df.empty:
             raise ValueError(f"No data found for {yf_ticker} on Yahoo")

        # Sync Directly
        sync_df_to_supabase(ticker, df)
        return df
        
    except Exception as e:
        raise ValueError(f"Yahoo fetch failed for {yf_ticker}: {e}")


def _get_supabase_last_date(ticker: str) -> Optional[dt.date]:
    """Helper to find the latest available date for a ticker in Supabase."""
    _init_supabase()
    if not supabase:
        return None
    try:
        sb_symbol = ticker
        sb_exchange = "US"
        if "." in ticker:
            parts = ticker.split(".")
            sb_symbol = parts[0]
            sb_exchange = parts[1]
            if sb_exchange == "CC": sb_exchange = "EGX"
            
        res = supabase.table("stock_prices")\
            .select("date")\
            .eq("symbol", sb_symbol)\
            .eq("exchange", sb_exchange)\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        if res.data:
            return pd.to_datetime(res.data[0]["date"]).date()
    except Exception as e:
        print(f"Error checking Supabase date for {ticker}: {e}")
    return None


def update_stock_data(
    api: APIClient,
    ticker: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    source: str = "eodhd",
    max_days: int = 365
) -> Tuple[bool, str]:
    """
    Updates stock data for a given ticker trying to be smart.
    source: 'eodhd' or 'yahoo'
    """
    print(f"DEBUG IN UPDATE: Ticker: {ticker}, Source received: {source}")
    if source.lower() != "eodhd":
        return False, "Only EODHD is supported for price updates"
    if api is None:
        return False, "EODHD API client is not configured"

    # cache_dir = _resolve_cache_dir(cache_dir) # Removed
    # _safe_mkdir(cache_dir) # Removed

    # existing_path = None # Removed
    # for p in _candidate_cache_paths(cache_dir, ticker): # Removed
    #     if os.path.exists(p): # Removed
    #         existing_path = p # Removed
    #         break # Removed
    # file_path = existing_path or _preferred_cache_path(cache_dir, ticker) # Removed
    # _safe_mkdir(os.path.dirname(file_path)) # Removed
    
    today = dt.date.today()
    
    # Disk-less: Check Supabase for last date
    last_date = _get_supabase_last_date(ticker)
    
    if last_date and last_date >= _last_trading_day(today):
         return True, "Already up to date in Cloud"

    # Check existing (from Supabase)
    existing_df = pd.DataFrame()
    # last_date = None # Now determined by _get_supabase_last_date
    
    # if os.path.exists(file_path): # Removed
    #     try:
    #         existing_df = pd.read_csv(file_path, index_col=0, parse_dates=True) # Removed
    #         # Cleanup: Remove invalid rows from legacy TradingView data
    #         if not existing_df.empty:
    #             # Convert index to datetime, coercing errors
    #             existing_df.index = pd.to_datetime(existing_df.index, errors='coerce')
    #             # Drop rows with NaT index (invalid dates)
    #             existing_df = existing_df[existing_df.index.notna()]
                
    #             if not existing_df.empty:
    #                 last_date = existing_df.index[-1].date()
    #                 if last_date >= _last_trading_day(today):
    #                  # Even if up to date, we might need to trim
    #                  if len(existing_df) > max_days:
    #                      trimmed = existing_df.tail(max_days)
    #                      trimmed.to_csv(file_path) # Removed
    #                      sync_local_to_supabase(ticker, file_path) # Removed
    #                      return True, f"Already up to date (Trimmed to {max_days} rows)"
    #                  return True, "Already up to date"
    #     except Exception:
    #         pass

    # EODHD Logic
    try:
        if last_date:
            # Append from last date
            start_date = last_date + dt.timedelta(days=1)
            new_df = api.get_eod_historical_stock_market_data(
                symbol=ticker, period="d", order="a", from_date=str(start_date)
            )
            new_df = _normalize_eodhd_eod_result(new_df)
            
            if new_df is None or new_df.empty:
                return True, "No new data (EODHD)"
            
            # Sync new rows directly
            sync_df_to_supabase(ticker, new_df)
            return True, f"Appended {len(new_df)} rows to Cloud"
            
        else:
             # Fresh Download
             start_date = today - dt.timedelta(days=max_days + 30) # Buffer
             df = api.get_eod_historical_stock_market_data(
                symbol=ticker, period="d", order="a", from_date=str(start_date)
            )
             df = _normalize_eodhd_eod_result(df)
             if df is None or df.empty:
                  return False, "No data (EODHD Fresh)"
             
             # Sync all directly
             sync_df_to_supabase(ticker, df)
             return True, f"Downloaded {len(df)} rows to Cloud"

    except Exception as e:
        return False, f"EODHD Error: {str(e)}"
             
    return True, "Data updated and synced"



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
            df = df.dropna(subset=['date']).sort_values('date').tail(1000) # Buffer
                
            rows = []
            sb_symbol = ticker
            sb_exchange = "US"
            if "." in ticker:
                parts = ticker.split(".")
                sb_symbol = parts[0]
                sb_exchange = parts[1]
                if sb_exchange == "CC": sb_exchange = "EGX"
            
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
                # Use chunks of 100 to avoid request size limits if needed
                for i in range(0, len(rows), 100):
                    supabase.table("stock_prices").upsert(rows[i:i+100], on_conflict="symbol,exchange,date").execute()
                print(f"DEBUG: Synced {len(rows)} rows for {sb_symbol}.{sb_exchange} to Supabase")
                return True, f"Synced {len(rows)} rows"
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
            if sb_exchange == "CC": sb_exchange = "EGX"

        row = {
            "symbol": sb_symbol,
            "exchange": sb_exchange,
            "data": data,
            "updated_at": dt.datetime.utcnow().isoformat()
        }
        
        supabase.table("stock_fundamentals").upsert(row, on_conflict="symbol,exchange").execute()
        return True, "Fundamentals synced"
    except Exception as e:
        msg = f"Supabase fund sync error for {ticker}: {e}"
        print(msg)
        return False, msg


def sync_fundamentals_to_supabase(ticker: str, file_path: str) -> tuple[bool, str]:
    """Read local JSON and upsert to Supabase stock_fundamentals."""
    _init_supabase()
    if not supabase:
        return False, "Supabase not initialized"
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            
        if isinstance(obj, dict) and "data" in obj:
            return sync_data_to_supabase(ticker, obj["data"])
        elif isinstance(obj, dict):
            return sync_data_to_supabase(ticker, obj)
            
        return False, "Invalid JSON format"
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
        # Supabase DELETE without .eq() or .match() will delete all rows if RLS/Permissions allow.
        # However, many clients require a filter. Using .neq("symbol", "") as a catch-all.
        res = supabase.table("stock_prices").delete().neq("symbol", "").execute()
        return True, "All stock prices cleared from Supabase"
    except Exception as e:
        msg = f"Error clearing stock prices: {e}"
        print(msg)
        return False, msg

            





def get_company_fundamentals(
    ticker: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    return_meta: bool = False,
    source: str = "auto",
) -> Any:
    cache_dir = _resolve_cache_dir(cache_dir)
    
    # 0. Try Supabase
    _init_supabase()
    if supabase:
        try:
            # Try to match symbol/exchange
            q = supabase.table("stock_fundamentals").select("data").eq("symbol", ticker)
            # If we had exchange metadata, we could filter better.
            res = q.limit(1).execute()
            if res.data and res.data[0].get("data"):
                data = res.data[0]["data"]
                return _ret(data, {"source": "supabase", "servedFrom": "supabase"})
        except Exception as e:
            print(f"Supabase fund read error: {e}")

    # Check cache
    cache_path = _preferred_fund_cache_path(cache_dir, ticker)
    _safe_mkdir(os.path.dirname(cache_path))

    fund_ttl_seconds = int(os.getenv("FUND_TTL_SECONDS", str(60 * 60 * 24 * 30)))
    fund_error_ttl_seconds = int(os.getenv("FUND_ERROR_TTL_SECONDS", str(60 * 60 * 6)))
    now_ts = int(dt.datetime.utcnow().timestamp())

    cached_data: Dict[str, Any] = {}
    cached_meta: Dict[str, Any] = {}

    # Normalize requested source early so cache logic can respect provider-specific errors.
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

    for cand_path in _candidate_fund_cache_paths(cache_dir, ticker):
        if not os.path.exists(cand_path):
            continue
        try:
            with open(cand_path, "r") as f:
                obj = json.load(f)

            if isinstance(obj, dict) and "data" in obj and "_meta" in obj:
                cached_data = obj.get("data") or {}
                cached_meta = obj.get("_meta") or {}
            elif isinstance(obj, dict):
                cached_data = obj
                cached_meta = {}

            if cached_data and not cached_meta:
                if _has_core_metrics(cached_data):
                    return _ret(cached_data, {"servedFrom": "cache_legacy"})
                cached_data = {}
                cached_meta = {}
                continue

            if cached_data:
                fetched_at = cached_meta.get("fetchedAt")
                if isinstance(fetched_at, int) and (now_ts - fetched_at) <= fund_ttl_seconds:
                    if _has_core_metrics(cached_data):
                        return _ret(cached_data, {**cached_meta, "servedFrom": "cache_fresh"})
                    cached_data = {}
                    cached_meta = {}
                    continue

            if cached_meta.get("status") == "error":
                fetched_at = cached_meta.get("fetchedAt")
                error_source = str(cached_meta.get("source") or "").lower()
                if (
                    isinstance(fetched_at, int)
                    and (now_ts - fetched_at) <= fund_error_ttl_seconds
                    and error_source == src
                ):
                    # Only respect an error cache when it was produced by the same
                    # provider that is being requested now. This allows switching
                    # fundSource (e.g. from tradingview to mubasher) without being
                    # stuck on a previous provider's error.
                    return _ret({}, {**cached_meta, "servedFrom": "error_cached"})
                # Otherwise ignore this error cache and try the requested provider.
                continue
        except Exception:
            pass

    def _try_tradingview() -> Any:
        upper = (ticker or "").strip().upper()
        if not upper:
            return None
        if upper.endswith(".EGX") or upper.endswith(".CC"):
            return None

        # Try to use our new consolidated integration
        from tradingview_integration import fetch_tradingview_fundamentals_bulk, fetch_tradingview_prices
        
        # 1. Fetch Fundamentals (Single symbol bulk wrapper)
        bulk = fetch_tradingview_fundamentals_bulk([upper], cache_dir=cache_dir)
        if upper not in bulk:
            return None
        
        data, meta = bulk[upper]
        
        # 2. Check if price history exists, if not, fetch it too
        price_cache_exists = any(os.path.exists(p) for p in _candidate_cache_paths(cache_dir, upper))
        if not price_cache_exists:
            # Automatic price history fetch on fundamental request if using TradingView source
            fetch_tradingview_prices(upper, cache_dir=cache_dir)
            
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
            got = _try_eodhd()
            if got is not None:
                return got

            if cached_data:
                fetched_at = cached_meta.get("fetchedAt")
                stale = bool(isinstance(fetched_at, int) and (now_ts - fetched_at) > fund_ttl_seconds)
                return _ret(cached_data, {**cached_meta, "servedFrom": "cache_stale" if stale else "cache"})

            # User request: Do not save error to cache
            # try:
            #     out_path = _preferred_fund_cache_path(cache_dir, ticker)
            #     _safe_mkdir(os.path.dirname(out_path))
            #     with open(out_path, "w") as f:
            #         json.dump({"_meta": {"fetchedAt": now_ts, "status": "error", "source": "mubasher"}, "data": {}}, f)
            # except Exception:
            #     pass

            return _ret({}, {"fetchedAt": now_ts, "status": "error", "source": "mubasher", "servedFrom": "error"})

    if src in {"tradingview", "auto"}:
        got = _try_tradingview()
        if got is not None:
            return got

        if src == "tradingview":
            got = _try_eodhd()
            if got is not None:
                return got

            if cached_data:
                fetched_at = cached_meta.get("fetchedAt")
                stale = bool(isinstance(fetched_at, int) and (now_ts - fetched_at) > fund_ttl_seconds)
                return _ret(cached_data, {**cached_meta, "servedFrom": "cache_stale" if stale else "cache"})

            # User request: Do not save error to cache
            # try:
            #     out_path = _preferred_fund_cache_path(cache_dir, ticker)
            #     _safe_mkdir(os.path.dirname(out_path))
            #     with open(out_path, "w") as f:
            #         json.dump({"_meta": {"fetchedAt": now_ts, "status": "error", "source": "tradingview"}, "data": {}}, f)
            # except Exception:
            #     pass

            return _ret({}, {"fetchedAt": now_ts, "status": "error", "source": "tradingview", "servedFrom": "error"})

    if src == "eodhd" or src == "auto":
        got = _try_eodhd()
        if got is not None:
            return got

        if cached_data:
            fetched_at = cached_meta.get("fetchedAt")
            stale = bool(isinstance(fetched_at, int) and (now_ts - fetched_at) > fund_ttl_seconds)
            return _ret(cached_data, {**cached_meta, "servedFrom": "cache_stale" if stale else "cache"})

        try:
            out_path = _preferred_fund_cache_path(cache_dir, ticker)
            _safe_mkdir(os.path.dirname(out_path))
            with open(out_path, "w") as f:
                json.dump({"_meta": {"fetchedAt": now_ts, "status": "error", "source": "eodhd"}, "data": {}}, f)
        except Exception:
            pass

        return _ret({}, {"fetchedAt": now_ts, "status": "error", "source": "eodhd", "servedFrom": "error"})

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
    # out["SMA_200"] = out["Close"].rolling(window=200).mean() # Removed to prevent massive data drop on short histories
    
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
    
    # Do not dropna here, let prepare_for_ai handle it so we can drop columns if needed
    return out


def prepare_for_ai(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Next_Close"] = out["Close"].shift(-1)
    out["Target"] = (out["Next_Close"] > out["Close"]).astype(int)
    out = out.dropna().copy()
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
) -> Tuple[RandomForestClassifier, List[str], pd.DataFrame, pd.Series, float]:
    preset = (rf_preset or "").lower()

    if preset == "fast" and len(df) > 600:
        df = df.iloc[-600:].copy()

    # Possible predictors
    all_predictors = ["Close", "Volume", "SMA_50", "SMA_200", "RSI", "Momentum"]
    # Only use those that exist in the dataframe
    predictors = [p for p in all_predictors if p in df.columns]

    if len(df) < 120:
        raise ValueError("Not enough data to train. Need at least ~120 rows.")

    if preset == "fast":
        test_size = 60 if len(df) > 260 else max(40, int(len(df) * 0.2))
    else:
        test_size = 100 if len(df) > 400 else max(40, int(len(df) * 0.2))
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]

    min_split = min(100, max(2, int(len(train) * 0.2)))

    model_kwargs = _sanitize_rf_params(
        rf_params,
        preset=rf_preset,
        train_len=len(train),
        default_min_split=min_split,
    )

    model = RandomForestClassifier(**model_kwargs)
    model.fit(train[predictors], train["Target"])

    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)

    precision = precision_score(test["Target"], preds, zero_division=0)
    return model, predictors, test, preds, float(precision)


def run_pipeline(
    api_key: str,
    ticker: str,
    from_date: str = "2020-01-01",
    cache_dir: str = DEFAULT_CACHE_DIR,
    include_fundamentals: bool = True,
    tolerance_days: int = 0,
    exchange: Optional[str] = None,
    force_local: bool = False,
    rf_params: Optional[Dict[str, Any]] = None,
    rf_preset: Optional[str] = None,
) -> Dict[str, Any]:
    api = APIClient(api_key)

    last_error: Optional[Exception] = None
    prices_ai: Optional[pd.DataFrame] = None
    selected_symbol: Optional[str] = None
    last_candidate_len: Optional[int] = None

    for sym in _candidate_symbols(ticker, exchange):
        try:
            raw = get_stock_data_eodhd(
                api, 
                sym, 
                from_date=from_date, 
                cache_dir=cache_dir, 
                tolerance_days=tolerance_days, 
                exchange=exchange,
                force_local=force_local
            )
            feat = add_technical_indicators(raw)
            
            # 1. Try with all features
            candidate = prepare_for_ai(feat)
            
            # 2. If not enough data (likely due to SMA_200 NaNs on a short history), 
            #    drop SMA_200 and try again.
            if len(candidate) < 120 and "SMA_200" in feat.columns:
                print(f"Warning: Insufficient data for SMA_200 (rows={len(candidate)}). Retrying without it.")
                feat_reduced = feat.drop(columns=["SMA_200"])
                candidate = prepare_for_ai(feat_reduced)

            last_candidate_len = len(candidate)

            if len(candidate) >= 120:
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
                f"Insufficient historical data for {ticker}{suffix}. Need at least ~120 rows after feature engineering, got {last_candidate_len}."
            ) from last_error
        raise ValueError(
            f"Insufficient historical data or symbol not available on your API plan for {ticker}{suffix}. Try specifying exchange, e.g. AAPL.US"
        ) from last_error

    model, predictors, test_df, preds, precision = train_and_predict(prices_ai, rf_params=rf_params, rf_preset=rf_preset)

    last_row = prices_ai.iloc[[-1]][predictors]
    tomorrow_prediction = int(model.predict(last_row)[0])

    # Fill NaNs for indicators to avoid JSON serialization issues or frontend gaps
    # Forward fill then backward fill to cover edges
    cols_to_fill = ["EMA_50", "EMA_200", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "RSI", "Momentum"]
    for c in cols_to_fill:
        if c in prices_ai.columns:
            prices_ai[c] = prices_ai[c].ffill().bfill().fillna(0.0)

    last_close = float(prices_ai.iloc[-1]["Close"])
    last_date = str(pd.to_datetime(prices_ai.index[-1]).date())

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

    fundamentals = get_company_fundamentals(selected_symbol or ticker, cache_dir) if include_fundamentals else {}
    fundamentals = _sanitize_fundamentals(fundamentals) if include_fundamentals else {}
    fundamentals_error = None
    if include_fundamentals and not fundamentals:
        fundamentals_error = "Fundamentals not available from yfinance."

    return {
        "ticker": selected_symbol or ticker,
        "precision": precision,
        "tomorrowPrediction": tomorrow_prediction,
        "lastClose": last_close,
        "lastDate": last_date,
        "fundamentals": fundamentals,
        "fundamentalsError": fundamentals_error,
        "testPredictions": out_preds,
        "note": "Precision is the % of correct UP predictions in the test set. Higher is better.",
    }

# Alias for compatibility
get_stock_data = get_stock_data_eodhd
