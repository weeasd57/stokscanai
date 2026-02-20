import math
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


def normalize_binance_symbol(sym: str, default_quote: str = "USDT") -> str:
    s = (sym or "").strip().upper()
    if not s:
        return s
    # Strip suffixes like .BINANCE
    if "." in s:
        s = s.split(".")[0]
    # Accept BINANCE:BTCUSDT format
    if ":" in s:
        s = s.split(":", 1)[1]
    s = s.replace("-", "").replace("_", "")
    if "/" in s:
        base, quote = s.split("/", 1)
        base = base.strip()
        quote = quote.strip()
        if quote in {"USD", "USDT"}:
            # Fix for USDT/USD -> USDTUSD (avoids USDTUSDT error)
            if base in {"USD", "USDT"}:
                return f"{base}{quote}"
            return f"{base}{default_quote}"
        return f"{base}{quote}"
    if s.endswith("USD") and not s.endswith("USDT") and not s.startswith("USDT"):
        return f"{s[:-3]}{default_quote}"
    return s


def binance_interval_for_timeframe(timeframe: str) -> str:
    t = (timeframe or "").strip().lower()
    mapping = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "1hour": "1h",
        "2hour": "2h",
        "4hour": "4h",
        "6hour": "6h",
        "12hour": "12h",
        "1day": "1d",
        "1week": "1w",
    }
    if t in mapping:
        return mapping[t]
    # Also accept already-normalized values
    if t in {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"}:
        return t
    # Default: 1h
    return "1h"


def _to_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


# Binance Base URLs to bypass potential IP blocks (Common on Hugging Face/Cloud)
BINANCE_BASE_URLS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision" # Public data node
]

def _binance_get(endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Optional[Any]:
    """Helper to try multiple Binance base URLs until success."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    last_err = None
    for base in BINANCE_BASE_URLS:
        try:
            url = f"{base}/{endpoint.lstrip('/')}"
            res = requests.get(url, params=params, headers=headers, timeout=timeout)
            
            # Normal success
            if res.status_code == 200:
                return res.json()
            
            # Handle specific failure codes (like 451 geo-block or 403) by trying next endpoint
            if res.status_code in [403, 429, 451]:
                last_err = f"Endpoint {base} returned {res.status_code}"
                continue
                
            res.raise_for_status()
        except Exception as e:
            last_err = f"Endpoint {base} failed: {str(e)}"
            continue
            
    print(f"DEBUG: Binance fetch failed across all endpoints. Last error: {last_err}")
    return None

def fetch_binance_klines(symbol: str, interval: str, limit: int = 1000) -> List[List[Any]]:
    params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    data = _binance_get("api/v3/klines", params=params)
    return data if isinstance(data, list) else []


def fetch_binance_bars_df(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    bn_symbol = normalize_binance_symbol(symbol, default_quote="USDT")
    interval = binance_interval_for_timeframe(timeframe)
    klines = fetch_binance_klines(bn_symbol, interval, limit=int(limit))
    if not klines:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for k in klines:
        # Binance kline format:
        # [ open_time, open, high, low, close, volume, close_time, quote_asset_volume, trades, ...]
        if not isinstance(k, list) or len(k) < 6:
            continue
        ts = pd.to_datetime(int(k[0]), unit="ms", utc=True)
        rows.append(
            {
                "timestamp": ts,
                "open": _to_float(k[1]) or 0.0,
                "high": _to_float(k[2]) or 0.0,
                "low": _to_float(k[3]) or 0.0,
                "close": _to_float(k[4]) or 0.0,
                "volume": _to_float(k[5]) or 0.0,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df

def fetch_all_binance_symbols(quote_asset: Optional[str] = "USDT", limit: int = 0) -> List[str]:
    """
    Fetches trading symbols from Binance.
    If limit > 0, fetches 24hr ticker stats and returns top N by Quote Volume.
    Otherwise returns all trading symbols alphabetically.
    """
    try:
        # If we need top N, we must get ticker stats for volume sorting
        if limit > 0:
            data = _binance_get("api/v3/ticker/24hr", timeout=20)
            if not data or not isinstance(data, list):
                return []
            # data is list of dicts: {symbol, quoteVolume, ...}
            
            # Filter for quote asset (suffix)
            valid = []
            q_suffix = (quote_asset or "USDT").upper()
            
            for item in data:
                s = item.get("symbol", "")
                if not s.endswith(q_suffix):
                    continue
                # Base is symbol minus suffix
                base = s[:-len(q_suffix)]
                
                # Exclude unusual symbols/tokens if needed.
                # For now just standard filter.
                
                vol = float(item.get("quoteVolume", 0) or 0)
                valid.append((s, base, vol))
            
            # Sort by volume desc
            valid.sort(key=lambda x: x[2], reverse=True)
            
            # Take top N
            top_n = valid[:limit]
            
            # Format as BASE/QUOTE
            return [f"{x[1]}/{q_suffix}" for x in top_n]

        # Otherwise standard exchangeInfo (lighter weight if we just want all)
        data = _binance_get("api/v3/exchangeInfo", timeout=15)
        if not data or not isinstance(data, dict):
             return []
        
        symbols = []
        for s in data.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            
            base = s.get("baseAsset")
            quote = s.get("quoteAsset")
            
            if quote_asset and quote != quote_asset:
                continue
            
            symbols.append(f"{base}/{quote}")
        return sorted(symbols)
    except Exception as e:
        print(f"DEBUG: Error in fetch_all_binance_symbols: {e}")
        return []
