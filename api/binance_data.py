import math
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


def normalize_binance_symbol(sym: str, default_quote: str = "USDT") -> str:
    s = (sym or "").strip().upper()
    if not s:
        return s
    # Accept BINANCE:BTCUSDT format
    if ":" in s:
        s = s.split(":", 1)[1]
    s = s.replace("-", "").replace("_", "")
    if "/" in s:
        base, quote = s.split("/", 1)
        base = base.strip()
        quote = quote.strip()
        if quote in {"USD", "USDT"}:
            return f"{base}{default_quote}"
        return f"{base}{quote}"
    if s.endswith("USD") and not s.endswith("USDT"):
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


def fetch_binance_klines(symbol: str, interval: str, limit: int = 1000) -> List[List[Any]]:
    params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    res = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=15)
    res.raise_for_status()
    data = res.json()
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

