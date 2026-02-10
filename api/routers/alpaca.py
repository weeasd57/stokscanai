import os
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Literal
import pandas as pd
import time as _time
import math
import ssl
import requests
import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus, AssetClass, AssetExchange
import api.stock_ai as stock_ai
from api.stock_ai import (
    _is_missing_table_error,
    _raise_transient_supabase_unavailable,
    _supabase_read_with_retry,
    _supabase_upsert_with_retry
)
from api.alpaca_cache import load_cached_assets, load_cached_exchanges, write_market_cache

router = APIRouter(prefix="/alpaca", tags=["alpaca"])

IntradayTimeframe = Literal["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]

class AlpacaAccountInfo(BaseModel):
    account_number: str
    status: str
    crypto_status: str
    currency: str
    buying_power: str
    cash: str
    portfolio_value: str
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    created_at: str

class AlpacaAsset(BaseModel):
    symbol: str
    name: str
    exchange: str
    class_name: str
    status: str
    tradable: bool
    marginable: bool
    shortable: bool
    easy_to_borrow: bool
    fractionable: bool

class AlpacaPositionInfo(BaseModel):
    symbol: str
    qty: str
    side: Optional[str] = None
    market_value: Optional[str] = None
    unrealized_intraday_pl: Optional[str] = None
    unrealized_pl: Optional[str] = None

class SyncRequest(BaseModel):
    symbols: List[str]
    asset_class: Optional[Literal["us_equity", "crypto"]] = "us_equity"
    exchange: Optional[str] = None
    source: Optional[Literal["local", "live"]] = "local"

class CacheRequest(BaseModel):
    markets: Optional[List[Literal["us_equity", "crypto"]]] = None


class PriceSyncRequest(BaseModel):
    symbols: List[str] = []
    asset_class: Optional[Literal["us_equity", "crypto"]] = "us_equity"
    exchange: Optional[str] = None
    days: int = 365
    source: Optional[Literal["local", "live", "tradingview", "binance"]] = "local"
    timeframe: IntradayTimeframe = "1d"

class CryptoDeleteRequest(BaseModel):
    symbols: List[str]
    timeframe: IntradayTimeframe = "1h"

def get_alpaca_client():
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    
    if not api_key or not secret_key:
        raise HTTPException(status_code=400, detail="Alpaca credentials (ALPACA_API_KEY, ALPACA_SECRET_KEY) not found in .env")
    
    return TradingClient(api_key, secret_key, paper=paper)


def _get_alpaca_market_data_client(asset_class: str):
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise HTTPException(status_code=400, detail="Alpaca credentials (ALPACA_API_KEY, ALPACA_SECRET_KEY) not found in .env")

    try:
        if asset_class == "crypto":
            from alpaca.data.historical import CryptoHistoricalDataClient

            return CryptoHistoricalDataClient(api_key, secret_key)
        else:
            from alpaca.data.historical import StockHistoricalDataClient

            return StockHistoricalDataClient(api_key, secret_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alpaca market data client not available: {e}")




def _safe_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        dt = pd.to_datetime(value).to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def _get_symbol_bar_stats(
    table: str,
    symbol: str,
    exchange: str,
    timeframe: str
) -> Dict[str, Any]:
    stock_ai._init_supabase()
    if not stock_ai.supabase:
        return {"count": 0, "min_ts": None, "max_ts": None}

    try:
        base = stock_ai.supabase.table(table).select("ts", count="exact").eq("symbol", symbol).eq("exchange", exchange).eq("timeframe", timeframe)
        count_res = base.execute()
        total = int(count_res.count or 0)

        max_res = stock_ai.supabase.table(table).select("ts").eq("symbol", symbol).eq("exchange", exchange).eq("timeframe", timeframe).order("ts", desc=True).limit(1).execute()
        min_res = stock_ai.supabase.table(table).select("ts").eq("symbol", symbol).eq("exchange", exchange).eq("timeframe", timeframe).order("ts", desc=False).limit(1).execute()

        max_ts = _safe_dt((max_res.data or [{}])[0].get("ts")) if max_res.data else None
        min_ts = _safe_dt((min_res.data or [{}])[0].get("ts")) if min_res.data else None

        return {"count": total, "min_ts": min_ts, "max_ts": max_ts}
    except Exception:
        return {"count": 0, "min_ts": None, "max_ts": None}

def _get_intraday_range(
    symbols: List[str],
    exchange: Optional[str],
    timeframe: str
) -> Dict[str, Any]:
    stock_ai._init_supabase()
    if not stock_ai.supabase:
        return {"count": 0, "min_ts": None, "max_ts": None}
    try:
        base = stock_ai.supabase.table("stock_bars_intraday").select("ts", count="exact").eq("timeframe", timeframe)
        if exchange:
            base = base.eq("exchange", exchange)
        if symbols:
            base = base.in_("symbol", symbols)
        count_res = base.execute()
        total = int(count_res.count or 0)

        min_q = stock_ai.supabase.table("stock_bars_intraday").select("ts").eq("timeframe", timeframe)
        max_q = stock_ai.supabase.table("stock_bars_intraday").select("ts").eq("timeframe", timeframe)
        if exchange:
            min_q = min_q.eq("exchange", exchange)
            max_q = max_q.eq("exchange", exchange)
        if symbols:
            min_q = min_q.in_("symbol", symbols)
            max_q = max_q.in_("symbol", symbols)

        min_res = min_q.order("ts", desc=False).limit(1).execute()
        max_res = max_q.order("ts", desc=True).limit(1).execute()

        min_ts = _safe_dt((min_res.data or [{}])[0].get("ts")) if min_res.data else None
        max_ts = _safe_dt((max_res.data or [{}])[0].get("ts")) if max_res.data else None
        return {"count": total, "min_ts": min_ts, "max_ts": max_ts}
    except Exception:
        return {"count": 0, "min_ts": None, "max_ts": None}

def _get_daily_range(
    symbols: List[str],
    exchange: Optional[str]
) -> Dict[str, Any]:
    stock_ai._init_supabase()
    if not stock_ai.supabase:
        return {"count": 0, "min_dt": None, "max_dt": None}
    try:
        base = stock_ai.supabase.table("stock_prices").select("date", count="exact")
        if exchange:
            base = base.eq("exchange", exchange)
        if symbols:
            base = base.in_("symbol", symbols)
        count_res = base.execute()
        total = int(count_res.count or 0)

        min_q = stock_ai.supabase.table("stock_prices").select("date")
        max_q = stock_ai.supabase.table("stock_prices").select("date")
        if exchange:
            min_q = min_q.eq("exchange", exchange)
            max_q = max_q.eq("exchange", exchange)
        if symbols:
            min_q = min_q.in_("symbol", symbols)
            max_q = max_q.in_("symbol", symbols)

        min_res = min_q.order("date", desc=False).limit(1).execute()
        max_res = max_q.order("date", desc=True).limit(1).execute()

        min_dt = _safe_dt((min_res.data or [{}])[0].get("date")) if min_res.data else None
        max_dt = _safe_dt((max_res.data or [{}])[0].get("date")) if max_res.data else None
        return {"count": total, "min_dt": min_dt, "max_dt": max_dt}
    except Exception:
        return {"count": 0, "min_dt": None, "max_dt": None}


def _dedupe_rows(rows: List[Dict[str, Any]], key_fields: List[str]) -> tuple[List[Dict[str, Any]], int]:
    """
    Remove duplicates that would violate an UPSERT's on_conflict constraint within the same command.
    Keeps the last occurrence.
    Returns (deduped_rows, duplicates_removed).
    """
    if not rows:
        return [], 0
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = "|".join(str(r.get(f, "")) for f in key_fields)
        out[k] = r
    deduped = list(out.values())
    return deduped, max(0, len(rows) - len(deduped))


def _first_not_none(obj: Any, names: List[str]) -> Any:
    for n in names:
        try:
            v = getattr(obj, n)
        except Exception:
            v = None
        if v is not None:
            return v
    return None


def _first_not_none_in_row(row: Any, names: List[str]) -> Any:
    for n in names:
        try:
            v = row.get(n)  # pandas Series or dict-like
        except Exception:
            v = None
        if v is not None:
            return v
    return None


def _to_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        import numpy as np  # type: ignore

        if isinstance(v, np.generic):
            v = v.item()
    except Exception:
        pass

    try:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            if math.isnan(v):
                return None
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            f = float(s)
            if math.isnan(f):
                return None
            return int(f)
        f = float(v)
        if math.isnan(f):
            return None
        return int(f)
    except Exception:
        return None


def _normalize_tv_symbol(sym: str, default_quote: str = "USDT") -> str:
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
    # If ends with USD and not USDT, switch to USDT
    if s.endswith("USD") and not s.endswith("USDT"):
        return f"{s[:-3]}{default_quote}"
    return s


def _tv_interval_for_timeframe(timeframe: str):
    from tvDatafeed import Interval
    mapping = {
        "1m": getattr(Interval, "in_1_minute", None),
        "5m": getattr(Interval, "in_5_minute", None),
        "15m": getattr(Interval, "in_15_minute", None),
        "30m": getattr(Interval, "in_30_minute", None),
        "1h": getattr(Interval, "in_1_hour", None),
        "2h": getattr(Interval, "in_2_hour", None),
        "4h": getattr(Interval, "in_4_hour", None),
        "1d": getattr(Interval, "in_daily", None),
        "1w": getattr(Interval, "in_weekly", None),
    }
    interval = mapping.get(timeframe)
    if interval is None:
        raise ValueError("Unsupported timeframe for TradingView")
    return interval

def _binance_interval_for_timeframe(timeframe: str) -> str:
    allowed = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"}
    if timeframe in allowed:
        return timeframe
    raise ValueError("Unsupported timeframe for Binance")

def _timeframe_seconds(timeframe: str) -> int:
    tf = (timeframe or "").strip().lower()
    if not tf:
        return 3600
    if tf.endswith("m"):
        return max(1, int(tf[:-1])) * 60
    if tf.endswith("h"):
        return max(1, int(tf[:-1])) * 3600
    if tf.endswith("d"):
        return max(1, int(tf[:-1] or "1")) * 86400
    if tf.endswith("w"):
        return max(1, int(tf[:-1] or "1")) * 7 * 86400
    return 3600

def _binance_fetch_klines(symbol: str, interval: str, limit: int = 1000, end_time_ms: Optional[int] = None) -> List[List[Any]]:
    params: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if end_time_ms is not None:
        params["endTime"] = end_time_ms
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    res = requests.get("https://api.binance.com/api/v3/klines", params=params, headers=headers, timeout=15)
    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Binance API error ({res.status_code}): {res.text}")
    data = res.json()
    if not isinstance(data, list):
        return []
    return data

def _binance_fetch_history(symbol: str, interval: str, target_bars: int) -> List[List[Any]]:
    collected: List[List[Any]] = []
    end_time_ms: Optional[int] = None
    while len(collected) < target_bars:
        limit = min(1000, target_bars - len(collected))
        chunk = _binance_fetch_klines(symbol, interval, limit=limit, end_time_ms=end_time_ms)
        if not chunk:
            break
        collected = chunk + collected
        first_open = chunk[0][0]
        if end_time_ms is not None and first_open >= end_time_ms:
            break
        end_time_ms = int(first_open) - 1
    return collected

@router.get("/account", response_model=AlpacaAccountInfo)
def get_account_status():
    try:
        client = get_alpaca_client()
        acc = client.get_account()
        return {
            "account_number": acc.account_number,
            "status": acc.status,
            "crypto_status": acc.crypto_status,
            "currency": acc.currency,
            "buying_power": acc.buying_power or "0",
            "cash": acc.cash or "0",
            "portfolio_value": acc.portfolio_value or "0",
            "pattern_day_trader": acc.pattern_day_trader,
            "trading_blocked": acc.trading_blocked,
            "transfers_blocked": acc.transfers_blocked,
            "account_blocked": acc.account_blocked,
            "created_at": str(acc.created_at)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/positions", response_model=List[AlpacaPositionInfo])
def get_positions():
    try:
        client = get_alpaca_client()
        positions = client.get_all_positions()
        out: List[Dict[str, Any]] = []
        for p in positions:
            out.append(
                {
                    "symbol": str(getattr(p, "symbol", "")),
                    "qty": str(getattr(p, "qty", "0")),
                    "side": str(getattr(p, "side", "")) if getattr(p, "side", None) is not None else None,
                    "market_value": str(getattr(p, "market_value", "")) if getattr(p, "market_value", None) is not None else None,
                    "unrealized_intraday_pl": str(getattr(p, "unrealized_intraday_pl", ""))
                    if getattr(p, "unrealized_intraday_pl", None) is not None
                    else None,
                    "unrealized_pl": str(getattr(p, "unrealized_pl", "")) if getattr(p, "unrealized_pl", None) is not None else None,
                }
            )
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/exchanges")
def get_exchanges(
    asset_class: Literal["us_equity", "crypto"] = Query("us_equity", description="us_equity or crypto"),
    source: Literal["local", "live"] = Query("local", description="local (cached) or live (Alpaca API)"),
):
    try:
        if source == "local":
            cached = load_cached_exchanges(asset_class)
            if cached is not None:
                return cached

        client = get_alpaca_client()
        search_params = GetAssetsRequest(
            status=AssetStatus.ACTIVE,
            asset_class=AssetClass.US_EQUITY if asset_class == "us_equity" else AssetClass.CRYPTO,
        )
        active_assets = client.get_all_assets(search_params)
        
        exchanges = {}
        for asset in active_assets:
            ex = str(asset.exchange.value) if hasattr(asset.exchange, 'value') else str(asset.exchange)
            exchanges[ex] = exchanges.get(ex, 0) + 1
            
        return [
            {"name": ex, "asset_count": count}
            for ex, count in sorted(exchanges.items())
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supabase-stats")
def get_alpaca_supabase_stats(
    asset_class: Literal["us_equity", "crypto"] = Query("us_equity", description="us_equity or crypto"),
    exchange: Optional[str] = Query(None, description="Optional Alpaca exchange filter (e.g., NASDAQ, NYSE, CRYPTO)"),
):
    """
    Returns a lightweight summary of what Alpaca-related data exists in Supabase.
    This is used by the Admin 'Data Center' UI.
    """
    stock_ai._init_supabase()
    sb = stock_ai.supabase
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not initialized")

    exchange_u = exchange.strip().upper() if isinstance(exchange, str) and exchange.strip() else None

    # Determine which exchanges to consider as "Alpaca" for us_equity.
    alpaca_exchanges: List[str] = []
    if asset_class == "crypto":
        alpaca_exchanges = ["CRYPTO"]
    else:
        cached = load_cached_exchanges("us_equity") or []
        alpaca_exchanges = [str(x.get("name") or "").strip().upper() for x in cached if (x.get("name") or "").strip()]
        if not alpaca_exchanges:
            alpaca_exchanges = ["NASDAQ", "NYSE", "ARCA", "BATS", "AMEX", "OTC", "EX"]

    if exchange_u:
        # If user passes exchange, use that explicitly.
        alpaca_exchanges = [exchange_u]

    # alpaca_assets_cache
    missing_tables: List[str] = []
    assets_count = 0
    assets_updated_at: Optional[str] = None
    try:
        def _fetch_assets(sb):
            q_count = sb.table("alpaca_assets_cache").select("symbol", count="exact").eq("asset_class", asset_class)
            q_latest = sb.table("alpaca_assets_cache").select("updated_at").eq("asset_class", asset_class)
            if exchange_u:
                q_count = q_count.eq("exchange", exchange_u)
                q_latest = q_latest.eq("exchange", exchange_u)
            
            c_res = q_count.limit(1).execute()
            l_res = q_latest.order("updated_at", desc=True).limit(1).execute()
            return c_res, l_res

        assets_count_res, assets_latest_res = _supabase_read_with_retry(_fetch_assets, table_name="alpaca_assets_cache")
        assets_count = int(assets_count_res.count or 0)
        if assets_latest_res.data and assets_latest_res.data[0].get("updated_at") is not None:
            assets_updated_at = str(assets_latest_res.data[0].get("updated_at"))
    except HTTPException as he:
        if isinstance(he.detail, dict) and he.detail.get("error") == "missing_table":
            missing_tables.append("alpaca_assets_cache")
        else:
            raise
    except Exception:
        raise

    # stock_prices (daily)
    daily_last_date: Optional[str] = None
    try:
        def _fetch_daily(sb):
            c_q = sb.table("stock_prices").select("symbol", count="exact").in_("exchange", alpaca_exchanges)
            l_q = sb.table("stock_prices").select("date").in_("exchange", alpaca_exchanges).order("date", desc=True)
            return c_q.limit(1).execute(), l_q.limit(1).execute()

        daily_count_res, daily_last_res = _supabase_read_with_retry(_fetch_daily, table_name="stock_prices")
        daily_count = int(daily_count_res.count or 0)
        if daily_last_res.data and daily_last_res.data[0].get("date") is not None:
            daily_last_date = str(daily_last_res.data[0].get("date"))
    except HTTPException as he:
        daily_count = 0
        daily_last_date = None
        if isinstance(he.detail, dict) and he.detail.get("error") == "missing_table":
            missing_tables.append("stock_prices")
        else:
            raise
    except Exception:
        daily_count = 0
        daily_last_date = None
        raise

    # stock_bars_intraday (intraday + crypto 1d bars)
    intraday_last_ts: Optional[str] = None
    intraday_total = 0
    try:
        def _fetch_intraday(sb):
            c_q = sb.table("stock_bars_intraday").select("symbol", count="exact").in_("exchange", alpaca_exchanges)
            l_q = sb.table("stock_bars_intraday").select("ts").in_("exchange", alpaca_exchanges).order("ts", desc=True)
            return c_q.limit(1).execute(), l_q.limit(1).execute()

        intraday_total_res, intraday_last_res = _supabase_read_with_retry(_fetch_intraday, table_name="stock_bars_intraday")
        intraday_total = int(intraday_total_res.count or 0)
        if intraday_last_res.data and intraday_last_res.data[0].get("ts") is not None:
            intraday_last_ts = str(intraday_last_res.data[0].get("ts"))
    except HTTPException as he:
        intraday_total = 0
        intraday_last_ts = None
        if isinstance(he.detail, dict) and he.detail.get("error") == "missing_table":
            if "stock_bars_intraday" not in missing_tables:
                missing_tables.append("stock_bars_intraday")
        else:
            raise
    except Exception:
        intraday_total = 0
        intraday_last_ts = None
        raise

    by_tf: Dict[str, int] = {}
    for tf in ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]:
        by_tf[tf] = 0
        try:
            def _fetch_tf(sb, timeframe=tf):
                return sb.table("stock_bars_intraday").select("symbol", count="exact").in_("exchange", alpaca_exchanges).eq("timeframe", timeframe).limit(1).execute()
            
            r_tf = _supabase_read_with_retry(lambda sb: _fetch_tf(sb, tf), table_name="stock_bars_intraday")
            by_tf[tf] = int(r_tf.count or 0)
        except HTTPException as he:
            if isinstance(he.detail, dict) and he.detail.get("error") == "missing_table":
                if "stock_bars_intraday" not in missing_tables:
                    missing_tables.append("stock_bars_intraday")
            else:
                raise
        except Exception:
            raise

    return {
        "asset_class": asset_class,
        "exchange_filter": exchange_u,
        "alpaca_exchanges": alpaca_exchanges,
        "missing_tables": missing_tables,
        "alpaca_assets_cache": {
            "rows": assets_count,
            "last_updated_at": assets_updated_at,
        },
        "stock_prices": {
            "rows": daily_count,
            "last_date": daily_last_date,
        },
        "stock_bars_intraday": {
            "rows": intraday_total,
            "last_ts": intraday_last_ts,
            "by_timeframe": by_tf,
        },
    }

@router.get("/crypto-symbols-stats")
def get_crypto_symbols_stats(
    timeframe: IntradayTimeframe = Query("1h", description="Timeframe for crypto stats"),
):
    stock_ai._init_supabase()
    sb = stock_ai.supabase
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not initialized")

    try:
        res = sb.rpc("get_crypto_symbol_stats", {"p_timeframe": timeframe}).execute()
        # Filter for /USD symbols only as per user request
        data = res.data or []
        filtered = [d for d in data if str(d.get("symbol", "")).strip().upper().endswith("/USD")]
        return filtered
    except Exception as e:
        if _is_missing_table_error(e, "stock_bars_intraday"):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "missing_table",
                    "table": "stock_bars_intraday",
                    "hint": "Run the SQL migration in supabase/schema.sql to create this table.",
                    "cause": str(e),
                },
            )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crypto-delete-bars")
def delete_crypto_bars(req: CryptoDeleteRequest):
    stock_ai._init_supabase()
    sb = stock_ai.supabase
    if not sb:
        raise HTTPException(status_code=500, detail="Supabase not initialized")

    symbols = [str(s or "").strip().upper() for s in (req.symbols or []) if str(s or "").strip()]
    if not symbols:
        raise HTTPException(status_code=422, detail="symbols required")

    timeframe = (req.timeframe or "1h").strip().lower()
    if timeframe not in {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"}:
        raise HTTPException(
            status_code=422,
            detail="timeframe must be one of: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w",
        )

    deleted_total = 0
    try:
        chunk_size = 100
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            res = (
                sb.table("stock_bars_intraday")
                .delete()
                .eq("exchange", "CRYPTO")
                .eq("timeframe", timeframe)
                .in_("symbol", chunk)
                .execute()
            )
            deleted_total += len(res.data or [])
        return {"success": True, "deleted": deleted_total, "symbols": len(symbols), "timeframe": timeframe}
    except Exception as e:
        if _is_missing_table_error(e, "stock_bars_intraday"):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "missing_table",
                    "table": "stock_bars_intraday",
                    "hint": "Run the SQL migration in supabase/schema.sql to create this table.",
                    "cause": str(e),
                },
            )
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assets", response_model=List[AlpacaAsset])
def get_assets(
    exchange: Optional[str] = Query(None, description="Exchange filter: NASDAQ, NYSE, ARCA, OTC"),
    status: str = Query("active", description="active or inactive"),
    asset_class: Literal["us_equity", "crypto"] = Query("us_equity", description="us_equity or crypto"),
    source: Literal["local", "live"] = Query("local", description="local (cached) or live (Alpaca API)"),
):
    try:
        if source == "local":
            cached_assets = load_cached_assets(asset_class, exchange=exchange)
            if cached_assets is not None:
                return cached_assets

        client = get_alpaca_client()

        ex_enum = None
        if exchange and asset_class == "us_equity":
            try:
                ex_enum = AssetExchange[exchange.upper()]
            except KeyError:
                ex_enum = None

        ac = AssetClass.US_EQUITY if asset_class == "us_equity" else AssetClass.CRYPTO
        search_params = GetAssetsRequest(
            status=AssetStatus.ACTIVE if status == "active" else AssetStatus.INACTIVE,
            asset_class=ac,
        )

        assets = client.get_all_assets(search_params)

        # Filter by exchange if requested
        if ex_enum is not None:
            assets = [a for a in assets if a.exchange == ex_enum]
        elif exchange:
            ex_low = exchange.strip().lower()
            assets = [
                a
                for a in assets
                if (str(a.exchange.value) if hasattr(a.exchange, "value") else str(a.exchange)).strip().lower() == ex_low
            ]

        return [
            {
                "symbol": a.symbol,
                "name": a.name,
                "exchange": str(a.exchange.value) if hasattr(a.exchange, "value") else str(a.exchange),
                "class_name": str(a.asset_class.value) if hasattr(a.asset_class, "value") else str(a.asset_class),
                "status": str(a.status.value) if hasattr(a.status, "value") else str(a.status),
                "tradable": bool(a.tradable),
                "marginable": bool(getattr(a, "marginable", False)),
                "shortable": bool(getattr(a, "shortable", False)),
                "easy_to_borrow": bool(getattr(a, "easy_to_borrow", False)),
                "fractionable": bool(getattr(a, "fractionable", False)),
            }
            for a in assets
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache-meta")
def get_cache_meta(
    asset_class: Literal["us_equity", "crypto"] = Query("us_equity", description="us_equity or crypto"),
):
    """
    Returns local cache metadata from `alpaca_exchanges/<asset_class>/meta.json`.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        meta_path = os.path.join(base_dir, "alpaca_exchanges", asset_class, "meta.json")
        if not os.path.exists(meta_path):
            return {"exists": False, "asset_class": asset_class}
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            meta = {}
        meta["exists"] = True
        meta["asset_class"] = asset_class
        return meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync")
def sync_alpaca_symbols(req: SyncRequest):
    """
    Saves Alpaca assets into Supabase:
      - Always upserts to `alpaca_assets_cache` (downloaded symbols)
      - Also upserts to `stock_fundamentals` (for scanner visibility)

    If `symbols` is empty, it saves all assets for the provided filters.
    """
    stock_ai._init_supabase()
    if not stock_ai.supabase:
        raise HTTPException(status_code=500, detail="Supabase not initialized")
    
    try:
        asset_class = (req.asset_class or "us_equity").strip().lower()
        if asset_class not in {"us_equity", "crypto"}:
            raise HTTPException(status_code=422, detail="asset_class must be us_equity or crypto")

        source = (req.source or "local").strip().lower()
        if source not in {"local", "live", "tradingview", "binance"}:
            raise HTTPException(status_code=422, detail="source must be local, live, tradingview, or binance")

        exchange_filter = (req.exchange or "").strip()

        # Prefer local cache for speed (fallback to live if missing)
        assets_list: Optional[List[Dict[str, Any]]] = None
        if source == "local":
            assets_list = load_cached_assets(asset_class, exchange=exchange_filter or None)
            if assets_list is None and exchange_filter:
                # If exchange file isn't present, try the full cache and filter in-memory
                assets_list = load_cached_assets(asset_class, exchange=None)

        if assets_list is None:
            client = get_alpaca_client()
            ac = AssetClass.US_EQUITY if asset_class == "us_equity" else AssetClass.CRYPTO
            live_assets = client.get_all_assets(GetAssetsRequest(asset_class=ac))
            assets_list = [
                {
                    "symbol": a.symbol,
                    "name": a.name,
                    "exchange": str(a.exchange.value) if hasattr(a.exchange, "value") else str(a.exchange),
                    "class_name": str(a.asset_class.value) if hasattr(a.asset_class, "value") else str(a.asset_class),
                    "status": str(a.status.value) if hasattr(a.status, "value") else str(a.status),
                    "tradable": bool(a.tradable),
                    "marginable": bool(getattr(a, "marginable", False)),
                    "shortable": bool(getattr(a, "shortable", False)),
                    "easy_to_borrow": bool(getattr(a, "easy_to_borrow", False)),
                    "fractionable": bool(getattr(a, "fractionable", False)),
                }
                for a in live_assets
            ]

        if exchange_filter:
            ex_low = exchange_filter.lower()
            assets_list = [a for a in (assets_list or []) if str(a.get("exchange") or "").lower() == ex_low]

        symbols = [s.strip().upper() for s in (req.symbols or []) if str(s or "").strip()]
        if symbols:
            sym_set = set(symbols)
            assets_list = [a for a in (assets_list or []) if str(a.get("symbol") or "").upper() in sym_set]

        if not assets_list:
            return {"success": True, "synced_count": 0, "saved_count": 0}

        now_iso = datetime.utcnow().isoformat()

        # 1) Save to alpaca_assets_cache (downloaded symbols table)
        cache_rows = []
        for a in assets_list:
            cache_rows.append(
                {
                    "symbol": a.get("symbol"),
                    "exchange": a.get("exchange") or "",
                    "asset_class": asset_class,
                    "name": a.get("name"),
                    "status": a.get("status"),
                    "tradable": a.get("tradable"),
                    "marginable": a.get("marginable"),
                    "shortable": a.get("shortable"),
                    "easy_to_borrow": a.get("easy_to_borrow"),
                    "fractionable": a.get("fractionable"),
                    "raw": a,
                    "updated_at": now_iso,
                }
            )

        chunk_size = 1000
        saved = 0
        try:
            for i in range(0, len(cache_rows), chunk_size):
                _supabase_upsert_with_retry(
                    "alpaca_assets_cache",
                    cache_rows[i : i + chunk_size],
                    on_conflict="symbol,exchange,asset_class",
                )
                saved += len(cache_rows[i : i + chunk_size])
        except Exception as e:
            if _is_missing_table_error(e, "alpaca_assets_cache"):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "missing_table",
                        "table": "alpaca_assets_cache",
                        "hint": "Run the SQL migration in supabase/schema.sql to create this table.",
                        "cause": str(e),
                    },
                )
            raise

        # 2) Also upsert into stock_fundamentals for scanner visibility
        fundamentals_rows = []
        for a in assets_list:
            ex = str(a.get("exchange") or "")
            fundamentals_rows.append(
                {
                    "symbol": a.get("symbol"),
                    "exchange": ex,
                    "data": {
                        "name": a.get("name"),
                        "symbol": a.get("symbol"),
                        "exchange": ex,
                        "country": "USA" if asset_class == "us_equity" else "CRYPTO",
                        "currency": "USD",
                        "type": "Common Stock" if asset_class == "us_equity" else "Crypto",
                    },
                }
            )

        try:
            for i in range(0, len(fundamentals_rows), chunk_size):
                _supabase_upsert_with_retry(
                    "stock_fundamentals",
                    fundamentals_rows[i : i + chunk_size],
                    on_conflict="symbol,exchange",
                )
        except Exception as e:
            if _is_missing_table_error(e, "stock_fundamentals"):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "missing_table",
                        "table": "stock_fundamentals",
                        "hint": "Run the SQL migration in supabase/schema.sql to create this table.",
                        "cause": str(e),
                    },
                )
            raise

        return {"success": True, "synced_count": len(fundamentals_rows), "saved_count": saved}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-local-cache")
def update_local_cache(req: CacheRequest = CacheRequest()):
    """
    Fetches assets from Alpaca and saves them locally under `alpaca_exchanges/`.
    Default markets: us_equity + crypto.
    """
    try:
        client = get_alpaca_client()
        markets = req.markets or ["us_equity", "crypto"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results: Dict[str, Any] = {"success": True, "timestamp": timestamp, "markets": {}}
        total_count = 0

        rows_for_db: List[Dict[str, Any]] = []

        for m in markets:
            ac = AssetClass.US_EQUITY if m == "us_equity" else AssetClass.CRYPTO
            assets = client.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=ac))

            api_assets = [
                {
                    "symbol": a.symbol,
                    "name": a.name,
                    "exchange": str(a.exchange.value) if hasattr(a.exchange, "value") else str(a.exchange),
                    "class_name": str(a.asset_class.value) if hasattr(a.asset_class, "value") else str(a.asset_class),
                    "status": str(a.status.value) if hasattr(a.status, "value") else str(a.status),
                    "tradable": bool(a.tradable),
                    "marginable": bool(getattr(a, "marginable", False)),
                    "shortable": bool(getattr(a, "shortable", False)),
                    "easy_to_borrow": bool(getattr(a, "easy_to_borrow", False)),
                    "fractionable": bool(getattr(a, "fractionable", False)),
                }
                for a in assets
            ]

            ts, count, exchanges = write_market_cache(m, api_assets, timestamp=timestamp)
            results["markets"][m] = {"count": count, "exchanges": exchanges, "updated_at": ts}
            total_count += int(count)

            for row in api_assets:
                rows_for_db.append(
                    {
                        "symbol": row.get("symbol"),
                        "exchange": row.get("exchange"),
                        "asset_class": m,
                        "name": row.get("name"),
                        "status": row.get("status"),
                        "tradable": row.get("tradable"),
                        "marginable": row.get("marginable"),
                        "shortable": row.get("shortable"),
                        "easy_to_borrow": row.get("easy_to_borrow"),
                        "fractionable": row.get("fractionable"),
                        "raw": row,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                )

        # Optional: upsert downloaded cache rows into Supabase
        stock_ai._init_supabase()
        if stock_ai.supabase and rows_for_db:
            try:
                chunk_size = 1000
                for i in range(0, len(rows_for_db), chunk_size):
                    _supabase_upsert_with_retry(
                        "alpaca_assets_cache",
                        rows_for_db[i : i + chunk_size],
                        on_conflict="symbol,exchange,asset_class",
                    )
                results["db_upserted"] = len(rows_for_db)
            except Exception as e:
                results["db_upserted_error"] = str(e)

        results["count"] = total_count
        results["filename"] = f"alpaca_exchanges/{timestamp}"
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync-prices")
def sync_alpaca_prices(req: PriceSyncRequest):
    """
    Downloads daily historical bars from Alpaca Market Data and upserts into Supabase `stock_prices`.

    If `symbols` is empty:
      - it will use the local cache (by asset_class/exchange) when `source=local`
      - otherwise it will pull the full live assets list (active) and sync all symbols
    """
    stock_ai._init_supabase()
    if not stock_ai.supabase:
        raise HTTPException(status_code=500, detail="Supabase not initialized")

    try:
        asset_class = (req.asset_class or "us_equity").strip().lower()
        if asset_class not in {"us_equity", "crypto"}:
            raise HTTPException(status_code=422, detail="asset_class must be us_equity or crypto")

        source = (req.source or "local").strip().lower()
        if source not in {"local", "live", "tradingview", "binance"}:
            raise HTTPException(status_code=422, detail="source must be local, live, tradingview, or binance")

        days = int(req.days or 0)
        if days < 5 or days > 5000:
            raise HTTPException(status_code=422, detail="days must be between 5 and 5000")

        timeframe = (req.timeframe or "1d").strip().lower()
        if timeframe not in {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"}:
            raise HTTPException(
                status_code=422,
                detail="timeframe must be one of: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w",
            )

        # Guardrails: minute/hour bars can explode row counts quickly.
        tf_seconds = _timeframe_seconds(timeframe)
        if timeframe == "1m" and days > 30:
            raise HTTPException(status_code=422, detail="For timeframe=1m, days must be <= 30")
        if tf_seconds < 3600 and days > 365:
            raise HTTPException(status_code=422, detail="For minute timeframes, days must be <= 365")
        if timeframe == "1h" and days > 3650:
            raise HTTPException(status_code=422, detail="For timeframe=1h, days must be <= 3650")

        exchange_filter = (req.exchange or "").strip()

        # Resolve symbol list
        symbols: List[str] = [s.strip().upper() for s in (req.symbols or []) if str(s or "").strip()]

        exchange_map: Dict[str, str] = {}
        if not symbols:
            assets_list = None
            if source == "local":
                assets_list = load_cached_assets(asset_class, exchange=exchange_filter or None)
                if assets_list is None and exchange_filter:
                    assets_list = load_cached_assets(asset_class, exchange=None)

            if assets_list is None:
                client = get_alpaca_client()
                ac = AssetClass.US_EQUITY if asset_class == "us_equity" else AssetClass.CRYPTO
                live_assets = client.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=ac))
                assets_list = [
                    {
                        "symbol": a.symbol,
                        "exchange": str(a.exchange.value) if hasattr(a.exchange, "value") else str(a.exchange),
                    }
                    for a in live_assets
                ]

            if exchange_filter:
                ex_low = exchange_filter.lower()
                assets_list = [a for a in (assets_list or []) if str(a.get("exchange") or "").lower() == ex_low]

            symbols = [str(a.get("symbol") or "").strip().upper() for a in (assets_list or []) if str(a.get("symbol") or "").strip()]
            exchange_map = {str(a.get("symbol") or "").strip().upper(): str(a.get("exchange") or "").strip() for a in (assets_list or [])}
        else:
            # If symbols are explicitly provided, still try to map exchange from local cache for consistent `stock_prices.exchange`
            cached_all = load_cached_assets(asset_class, exchange=None) or []
            exchange_map = {str(a.get("symbol") or "").strip().upper(): str(a.get("exchange") or "").strip() for a in cached_all}

        if not symbols:
            return {"success": True, "symbols": 0, "rows_upserted": 0}

        # Normalize to per-symbol rows
        rows: List[Dict[str, Any]] = []
        intraday_rows: List[Dict[str, Any]] = []
        volume_missing = 0
        volume_total = 0
        store_intraday = (timeframe != "1d") or (asset_class == "crypto")

        if source == "tradingview":
            if asset_class != "crypto":
                raise HTTPException(status_code=422, detail="tradingview source only supported for crypto")
            if timeframe == "1m" and days > 7:
                raise HTTPException(status_code=422, detail="For timeframe=1m with tradingview, days must be <= 7")

            try:
                from tvDatafeed import TvDatafeed
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"tvDatafeed not available: {e}")

            tv_exchange = exchange_filter or "BINANCE"
            try:
                interval = _tv_interval_for_timeframe(timeframe)
            except Exception as e:
                raise HTTPException(status_code=422, detail=str(e))

            tv = TvDatafeed()
            for sym in symbols:
                tv_symbol = _normalize_tv_symbol(sym, default_quote="USDT")
                if not tv_symbol:
                    continue
                desired_total = max(1, int(math.ceil((days * 86400) / max(1, tf_seconds))))

                stats = _get_symbol_bar_stats("stock_bars_intraday", sym.upper(), "CRYPTO", timeframe)
                existing_count = int(stats.get("count") or 0)
                max_ts = stats.get("max_ts")
                now_utc = datetime.now(timezone.utc)
                if max_ts:
                    delta = now_utc - max_ts
                    missing_forward = max(0, int(delta.total_seconds() // max(1, tf_seconds)))
                else:
                    missing_forward = desired_total

                remaining_needed = max(0, desired_total - (existing_count + missing_forward))
                n_bars = existing_count + missing_forward + remaining_needed + 50
                if timeframe == "1h":
                    n_bars = min(5000, max(desired_total, n_bars))
                elif timeframe == "1d":
                    n_bars = min(5000, max(desired_total, n_bars))
                else:
                    n_bars = min(5000, max(desired_total, n_bars))

                df = tv.get_hist(symbol=tv_symbol, exchange=tv_exchange, interval=interval, n_bars=n_bars)
                if df is None or df.empty:
                    continue
                df = df.reset_index()
                if "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "ts"})
                for _, r in df.iterrows():
                    ts = r.get("ts") or r.get("time") or r.get("timestamp")
                    if ts is None:
                        continue
                    try:
                        d = pd.to_datetime(ts).date().isoformat()
                    except Exception:
                        d = str(ts)[:10]
                    v_int = _to_int_or_none(r.get("volume"))
                    volume_total += 1
                    if v_int is None:
                        volume_missing += 1
                    ex_val = "CRYPTO"

                    open_v = r.get("open")
                    high_v = r.get("high")
                    low_v = r.get("low")
                    close_v = r.get("close")
                    rows.append(
                        {
                            "symbol": sym.strip().upper(),
                            "exchange": ex_val,
                            "date": d,
                            "open": float(open_v) if open_v is not None else None,
                            "high": float(high_v) if high_v is not None else None,
                            "low": float(low_v) if low_v is not None else None,
                            "close": float(close_v) if close_v is not None else None,
                            "adjusted_close": float(close_v) if close_v is not None else None,
                            "volume": v_int,
                        }
                    )
                    if store_intraday:
                        try:
                            ts_iso = pd.to_datetime(ts, utc=True).to_pydatetime().isoformat()
                        except Exception:
                            ts_iso = str(ts)
                        intraday_rows.append(
                            {
                                "symbol": sym.strip().upper(),
                                "exchange": ex_val,
                                "timeframe": timeframe,
                                "ts": ts_iso,
                                "open": float(open_v) if open_v is not None else None,
                                "high": float(high_v) if high_v is not None else None,
                                "low": float(low_v) if low_v is not None else None,
                                "close": float(close_v) if close_v is not None else None,
                                "volume": v_int,
                            }
                        )
        elif source == "binance":
            if asset_class != "crypto":
                raise HTTPException(status_code=422, detail="binance source only supported for crypto")
            try:
                interval = _binance_interval_for_timeframe(timeframe)
            except Exception as e:
                raise HTTPException(status_code=422, detail=str(e))

            desired_total = max(1, int(math.ceil((days * 86400) / max(1, tf_seconds))))

            for sym in symbols:
                bn_symbol = _normalize_tv_symbol(sym, default_quote="USDT")
                if not bn_symbol:
                    continue
                stats = _get_symbol_bar_stats("stock_bars_intraday", sym.upper(), "CRYPTO", timeframe)
                existing_count = int(stats.get("count") or 0)
                max_ts = stats.get("max_ts")
                now_utc = datetime.now(timezone.utc)
                if max_ts:
                    delta = now_utc - max_ts
                    missing_forward = max(0, int(delta.total_seconds() // max(1, tf_seconds)))
                else:
                    missing_forward = desired_total

                if existing_count < desired_total:
                    target_bars = desired_total + 50
                else:
                    target_bars = max(50, missing_forward + 50)
                try:
                    klines = _binance_fetch_history(bn_symbol, interval, target_bars)
                except Exception as e:
                    print(f"DEBUG: Binance fetch failed for {bn_symbol}: {e}")
                    # If only one symbol was requested, don't fail silently
                    if len(symbols) == 1:
                        raise HTTPException(status_code=500, detail=f"Failed to fetch {bn_symbol} from Binance: {str(e)}")
                    continue

                for k in klines:
                    try:
                        open_time_ms = int(k[0])
                    except Exception:
                        continue
                    ts = datetime.utcfromtimestamp(open_time_ms / 1000.0).replace(tzinfo=timezone.utc)
                    try:
                        d = ts.date().isoformat()
                    except Exception:
                        d = str(ts)[:10]
                    v_int = _to_int_or_none(k[5])
                    volume_total += 1
                    if v_int is None:
                        volume_missing += 1
                    ex_val = "CRYPTO"
                    open_v = k[1]
                    high_v = k[2]
                    low_v = k[3]
                    close_v = k[4]
                    rows.append(
                        {
                            "symbol": sym.strip().upper(),
                            "exchange": ex_val,
                            "date": d,
                            "open": float(open_v) if open_v is not None else None,
                            "high": float(high_v) if high_v is not None else None,
                            "low": float(low_v) if low_v is not None else None,
                            "close": float(close_v) if close_v is not None else None,
                            "adjusted_close": float(close_v) if close_v is not None else None,
                            "volume": v_int,
                        }
                    )
                    if store_intraday:
                        intraday_rows.append(
                            {
                                "symbol": sym.strip().upper(),
                                "exchange": ex_val,
                                "timeframe": timeframe,
                                "ts": ts.isoformat(),
                                "open": float(open_v) if open_v is not None else None,
                                "high": float(high_v) if high_v is not None else None,
                                "low": float(low_v) if low_v is not None else None,
                                "close": float(close_v) if close_v is not None else None,
                                "volume": v_int,
                            }
                        )
        else:
            # Date range
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=days)

            md_client = _get_alpaca_market_data_client(asset_class)

            # Build request
            try:
                from alpaca.data.timeframe import TimeFrame

                tf = TimeFrame.Day
                if timeframe == "1m":
                    tf = TimeFrame.Minute
                elif timeframe == "1h":
                    tf = TimeFrame.Hour

                if asset_class == "crypto":
                    from alpaca.data.requests import CryptoBarsRequest

                    bars_req = CryptoBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start_dt, end=end_dt)
                    resp = md_client.get_crypto_bars(bars_req)
                else:
                    from alpaca.data.requests import StockBarsRequest

                    bars_req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start_dt, end=end_dt)
                    resp = md_client.get_stock_bars(bars_req)
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": "alpaca_market_data_failed",
                        "asset_class": asset_class,
                        "symbols_count": len(symbols),
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat(),
                        "timeframe": timeframe,
                        "cause": str(e),
                    },
                )

            if hasattr(resp, "data") and isinstance(getattr(resp, "data"), dict):
                data_dict = resp.data  # type: ignore[assignment]
                for sym, bars in (data_dict or {}).items():
                    sym_u = str(sym or "").strip().upper()
                    ex_val = exchange_filter or exchange_map.get(sym_u) or ("CRYPTO" if asset_class == "crypto" else "ALPACA")
                    for b in (bars or []):
                        ts = getattr(b, "timestamp", None) or getattr(b, "t", None)
                        if ts is None:
                            continue
                        try:
                            d = pd.to_datetime(ts).date().isoformat()
                        except Exception:
                            d = str(ts)[:10]
                        o = _first_not_none(b, ["open", "o"])
                        h = _first_not_none(b, ["high", "h"])
                        l = _first_not_none(b, ["low", "l"])
                        c = _first_not_none(b, ["close", "c"])
                        v = _first_not_none(b, ["volume", "v"])
                        v_int = _to_int_or_none(v)

                        volume_total += 1
                        if v_int is None:
                            volume_missing += 1

                        rows.append(
                            {
                                "symbol": sym_u,
                                "exchange": str(ex_val),
                                "date": d,
                                "open": float(o) if o is not None else None,
                                "high": float(h) if h is not None else None,
                                "low": float(l) if l is not None else None,
                                "close": float(c) if c is not None else None,
                                "adjusted_close": float(c) if c is not None else None,
                                "volume": v_int,
                            }
                        )

                        # For 1m/1h, store full timestamp bars in intraday table.
                        if store_intraday:
                            try:
                                ts_iso = pd.to_datetime(ts, utc=True).to_pydatetime().isoformat()
                            except Exception:
                                ts_iso = str(ts)
                            intraday_rows.append(
                                {
                                    "symbol": sym_u,
                                    "exchange": str(ex_val),
                                    "timeframe": timeframe,
                                    "ts": ts_iso,
                                    "open": float(o) if o is not None else None,
                                    "high": float(h) if h is not None else None,
                                    "low": float(l) if l is not None else None,
                                    "close": float(c) if c is not None else None,
                                    "volume": v_int,
                                }
                            )
            elif hasattr(resp, "df"):
                # Fallback: pandas df with multi-index (symbol, timestamp)
                try:
                    df = resp.df  # type: ignore[attr-defined]
                    if df is not None and not df.empty:
                        # Ensure columns standard: open, high, low, close, volume
                        for idx, r in df.reset_index().iterrows():
                            sym_u = str(r.get("symbol") or r.get("Symbol") or "").strip().upper()
                            ts = r.get("timestamp") or r.get("time") or r.get("datetime") or r.get("t")
                            if not sym_u or ts is None:
                                continue
                            try:
                                d = pd.to_datetime(ts).date().isoformat()  # type: ignore[name-defined]
                            except Exception:
                                d = str(ts)[:10]
                            ex_val = exchange_filter or exchange_map.get(sym_u) or ("CRYPTO" if asset_class == "crypto" else "ALPACA")
                            v = _first_not_none_in_row(r, ["volume", "v"])
                            v_int = _to_int_or_none(v)

                            volume_total += 1
                            if v_int is None:
                                volume_missing += 1
                            open_v = _first_not_none_in_row(r, ["open", "o"])
                            high_v = _first_not_none_in_row(r, ["high", "h"])
                            low_v = _first_not_none_in_row(r, ["low", "l"])
                            close_v = _first_not_none_in_row(r, ["close", "c"])
                            rows.append(
                                {
                                    "symbol": sym_u,
                                    "exchange": str(ex_val),
                                    "date": d,
                                    "open": float(open_v) if open_v is not None else None,
                                    "high": float(high_v) if high_v is not None else None,
                                    "low": float(low_v) if low_v is not None else None,
                                    "close": float(close_v) if close_v is not None else None,
                                    "adjusted_close": float(close_v) if close_v is not None else None,
                                    "volume": v_int,
                                }
                            )

                            if store_intraday:
                                ts = r.get("timestamp") or r.get("time") or r.get("datetime") or r.get("t")
                                try:
                                    ts_iso = pd.to_datetime(ts, utc=True).to_pydatetime().isoformat()
                                except Exception:
                                    ts_iso = str(ts)
                                intraday_rows.append(
                                    {
                                        "symbol": sym_u,
                                        "exchange": str(ex_val),
                                        "timeframe": timeframe,
                                        "ts": ts_iso,
                                        "open": float(r.get("open") if r.get("open") is not None else r.get("o")) if (r.get("open") is not None or r.get("o") is not None) else None,
                                        "high": float(r.get("high") if r.get("high") is not None else r.get("h")) if (r.get("high") is not None or r.get("h") is not None) else None,
                                        "low": float(r.get("low") if r.get("low") is not None else r.get("l")) if (r.get("low") is not None or r.get("l") is not None) else None,
                                        "close": float(r.get("close") if r.get("close") is not None else r.get("c")) if (r.get("close") is not None or r.get("c") is not None) else None,
                                        "volume": v_int,
                                    }
                                )
                except Exception:
                    rows = []

        daily_dupes_removed = 0
        intraday_dupes_removed = 0
        if store_intraday:
            intraday_rows, intraday_dupes_removed = _dedupe_rows(intraday_rows, ["symbol", "exchange", "timeframe", "ts"])
            if not intraday_rows:
                return {"success": True, "symbols": len(symbols), "rows_upserted": 0, "timeframe": timeframe}

        if timeframe == "1d":
            rows, daily_dupes_removed = _dedupe_rows(rows, ["symbol", "exchange", "date"])
            if not rows:
                return {"success": True, "symbols": len(symbols), "rows_upserted": 0, "timeframe": timeframe}

        # Upsert in chunks
        chunk_size = 500 if _timeframe_seconds(timeframe) <= 3600 else 1000
        upserted = 0
        tables: List[str] = []
        try:
            if store_intraday and intraday_rows:
                for i in range(0, len(intraday_rows), chunk_size):
                    _supabase_upsert_with_retry(
                        "stock_bars_intraday",
                        intraday_rows[i : i + chunk_size],
                        on_conflict="symbol,exchange,timeframe,ts",
                    )
                    upserted += len(intraday_rows[i : i + chunk_size])
                tables.append("stock_bars_intraday")

            if timeframe == "1d" and rows:
                for i in range(0, len(rows), chunk_size):
                    _supabase_upsert_with_retry("stock_prices", rows[i : i + chunk_size])
                    upserted += len(rows[i : i + chunk_size])
                tables.append("stock_prices")
        except Exception as e:
            if _is_missing_table_error(e, "stock_bars_intraday") or _is_missing_table_error(e, "stock_prices"):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "missing_table",
                        "table": "stock_bars_intraday" if store_intraday else "stock_prices",
                        "hint": "Run the SQL migration in supabase/schema.sql to create this table.",
                        "cause": str(e),
                    },
                )
            raise

        range_exchange = None
        if asset_class == "crypto":
            range_exchange = "CRYPTO"
        elif exchange_filter:
            range_exchange = exchange_filter

        date_range: Dict[str, Any] = {"from": None, "to": None, "count": 0, "table": "stock_bars_intraday" if store_intraday else "stock_prices"}
        if store_intraday:
            r = _get_intraday_range([s.strip().upper() for s in symbols], range_exchange, timeframe)
            date_range["from"] = r.get("min_ts").isoformat() if r.get("min_ts") else None
            date_range["to"] = r.get("max_ts").isoformat() if r.get("max_ts") else None
            date_range["count"] = int(r.get("count") or 0)
        else:
            r = _get_daily_range([s.strip().upper() for s in symbols], range_exchange)
            date_range["from"] = r.get("min_dt").isoformat() if r.get("min_dt") else None
            date_range["to"] = r.get("max_dt").isoformat() if r.get("max_dt") else None
            date_range["count"] = int(r.get("count") or 0)

        return {
            "success": True,
            "symbols": len(symbols),
            "rows_upserted": upserted,
            "days": days,
            "timeframe": timeframe,
            "table": "stock_bars_intraday" if store_intraday and timeframe != "1d" else ("stock_bars_intraday" if asset_class == "crypto" else "stock_prices"),
            "tables": tables,
            "date_range": date_range,
            "volume_total": volume_total,
            "volume_missing": volume_missing,
            "duplicates_removed": {
                "intraday": intraday_dupes_removed,
                "daily": daily_dupes_removed,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
