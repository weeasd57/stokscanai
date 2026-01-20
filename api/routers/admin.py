import os
import json
import requests
from typing import List, Optional
from collections import defaultdict
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from eodhd import APIClient
from dotenv import load_dotenv
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from supabase import create_client, Client


from api.stock_ai import (
    get_stock_data, get_stock_data_eodhd, get_company_fundamentals,
    _init_supabase, supabase, update_stock_data,
    _finite_float, add_technical_indicators, upsert_technical_indicators,
)

router = APIRouter(prefix="/admin", tags=["admin"])

def _chunks(items: list, size: int):
    """Yield successive n-sized chunks from items."""
    for i in range(0, len(items), size):
        yield items[i : i + size]

class UpdateRequest(BaseModel):
    symbols: List[str]
    country: Optional[str] = None
    updatePrices: bool = True
    updateFundamentals: bool = False
    maxPriceDays: int = 365

class SyncRequest(BaseModel):
    exchange: Optional[str] = None # If None, sync all
    force: bool = False # If True, re-upload even if no change (not used yet)

class SmartSyncRequest(BaseModel):
    exchange: str
    days: int = 365
    updatePrices: bool = True
    updateFunds: bool = False
    unified: bool = False

class ScheduleDispatchRequest(BaseModel):
    workflow: str  # 'ai-training' or 'data-sync'
    when: str      # ISO datetime string (UTC or local; if naive, treated as UTC)
    exchange: Optional[str] = None
    days: Optional[int] = None
    updatePrices: Optional[bool] = None
    updateFunds: Optional[bool] = None
    unified: Optional[bool] = None

class LogoDownloadRequest(BaseModel):
    exchange: Optional[str] = None
    country: Optional[str] = None

class CronTriggerRequest(BaseModel):
    action: str # update_prices, update_funds, recalculate_technicals
    secret: str
    exchange: Optional[str] = None

class ScheduleRequest(BaseModel):
    cron: str
    startTime: str = "22:30"
    endTime: str = "04:00"



# Valid sources
PRICE_SOURCES = ["eodhd", "tradingview", "cache"]
FUND_SOURCES = ["auto", "mubasher", "tradingview", "eodhd"]

# Simple in-memory or file-based config
# The app runs from the project root.
CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "admin_config.json"))

ENV_ROOT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

supabase: Optional[Client] = None
def _init_supabase():
    global supabase
    if supabase is None:
        url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if url and key:
            supabase = create_client(url, key)


# Simple in-memory state to track local training status
LOCAL_TRAINING_STATE = {
    "running": False,
    "exchange": None,
    "started_at": None,
    "completed_at": None,
    "error": None,
    "last_message": None,
    "phase": None,
    "stats": {},
    "version": 0,
    "last_update": None,
}
_local_training_lock = threading.Lock()


@router.get("/plans")
def get_plans():
    _init_supabase()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not initialized")
    try:
        res = supabase.table("plans").select("*").order("price").execute()
        return res.data
    except Exception as e:
        print(f"Error fetching plans: {e}")
        # Return fallback plans if table doesn't exist yet to avoid breaking frontend
        return [
            {
                "name": "Free",
                "price": 0,
                "period": "forever",
                "desc": "For beginners exploring AI insights",
                "features": ["Daily AI Predictions (limited)", "Basic Technical Indicators", "Public Market Data", "Community Support"],
                "featured": False,
                "button_text": "Current Plan"
            },
            {
                "name": "Pro",
                "price": 29,
                "period": "month",
                "desc": "For serious traders and analysts",
                "features": ["Unlimited AI Predictions", "Advanced RandomForest Analysis", "Real-time Data Access", "Custom Watchlist Alerts", "Priority Support", "Early access to new features"],
                "featured": True,
                "button_text": "Go Pro"
            },
            {
                "name": "Enterprise",
                "price": 99,
                "period": "month",
                "desc": "For professional teams and hedge funds",
                "features": ["API Access (Rest & Websocket)", "Custom Model Training", "Bulk Data Exports", "White-label Reports", "Dedicated Account Manager", "SLA Guarantee"],
                "featured": False,
                "button_text": "Contact Sales"
            }
        ]

def _reload_env() -> None:
    load_dotenv(ENV_ROOT_FILE, override=True)

def _load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
                if not isinstance(cfg, dict):
                    cfg = {}

                legacy = cfg.get("source")
                if legacy and "priceSource" not in cfg:
                    cfg["priceSource"] = legacy

                price_source = cfg.get("priceSource") or "eodhd"
                fund_source = cfg.get("fundSource") or "auto"
                max_workers = cfg.get("maxWorkers")

                if price_source == "cache":
                    price_source = "tradingview"

                if price_source not in PRICE_SOURCES:
                    price_source = "eodhd"
                if fund_source not in FUND_SOURCES:
                    fund_source = "auto"
                if not isinstance(max_workers, int) or max_workers <= 0:
                    max_workers = int(os.getenv("ADMIN_MAX_WORKERS", "8"))

                return {
                    "source": price_source,
                    "priceSource": price_source,
                    "fundSource": fund_source,
                    "maxWorkers": max_workers,
                }
        except Exception:
            pass
    max_workers = int(os.getenv("ADMIN_MAX_WORKERS", "8"))
    return {"source": "eodhd", "priceSource": "eodhd", "fundSource": "auto", "maxWorkers": max_workers}

def _save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)

@router.get("/config")
def get_config():
    return _load_config()

class ConfigUpdate(BaseModel):
    priceSource: Optional[str] = None
    fundSource: Optional[str] = None
    maxWorkers: Optional[int] = None

@router.post("/config")
def set_config(cfg: ConfigUpdate):
    current = _load_config()

    if cfg.priceSource is not None:
        incoming = cfg.priceSource
        if incoming == "cache":
            incoming = "tradingview"
        if incoming not in PRICE_SOURCES:
            raise HTTPException(status_code=400, detail="Invalid priceSource")
        current["priceSource"] = incoming
        current["source"] = incoming

    if cfg.fundSource is not None:
        if cfg.fundSource not in FUND_SOURCES:
            raise HTTPException(status_code=400, detail="Invalid fundSource")
        current["fundSource"] = cfg.fundSource

    if cfg.maxWorkers is not None:
        if not isinstance(cfg.maxWorkers, int) or cfg.maxWorkers <= 0:
            raise HTTPException(status_code=400, detail="Invalid maxWorkers")
        current["maxWorkers"] = cfg.maxWorkers

    _save_config(current)
    return current


@router.get("/fundamentals/{ticker}")
def get_fundamentals(ticker: str, source: Optional[str] = None):
    _reload_env()
    cfg = _load_config()
    selected = (source or cfg.get("fundSource") or "auto")
    data, meta = get_company_fundamentals(ticker, return_meta=True, source=selected)
    return {"ticker": ticker, "data": data, "meta": meta}

@router.post("/update_batch")
def update_batch(req: UpdateRequest, background_tasks: BackgroundTasks):
    _reload_env()
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        # Only strict if using eodhd?
        # For now, let's keep it safe.
        pass

    config = _load_config()
    price_source = config.get("priceSource") or config.get("source") or "eodhd"
    fund_source_cfg = config.get("fundSource") or "auto"
    max_workers = config.get("maxWorkers")
    if not isinstance(max_workers, int) or max_workers <= 0:
        max_workers = int(os.getenv("ADMIN_MAX_WORKERS", "8"))

    if price_source == "eodhd" and not api_key:
        raise HTTPException(status_code=500, detail="EODHD_API_KEY not set")

    client = APIClient(api_key) if api_key else None
    results: List[dict] = []
    unauthorized = False
    unauthorized_lock = threading.Lock()

    def _trim_fund_data(d: dict) -> dict:
        if not isinstance(d, dict):
            return {}
        keys = [
            "marketCap",
            "peRatio",
            "eps",
            "dividendYield",
            "high52",
            "low52",
            "beta",
            "sector",
            "industry",
            "name",
            "country",
            "currency",
        ]
        out = {}
        for k in keys:
            if k in d:
                out[k] = d.get(k)
        return out

    def _has_core_fund_metrics(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        core = ["marketCap", "peRatio", "eps", "dividendYield", "beta", "high52", "low52"]
        for k in core:
            if d.get(k) is not None:
                return True
        return False

    def _tradingview_market_for_symbol(sym: str) -> str:
        """Get TradingView market name from symbol - wrapper for module function."""
        from tradingview_integration import get_tradingview_market
        return get_tradingview_market(sym)

    def _bulk_tradingview_fundamentals(tickers: List[str]) -> dict:
        """Bulk fetch fundamentals from TradingView - wrapper for module function."""
        from tradingview_integration import fetch_tradingview_fundamentals_bulk
        return fetch_tradingview_fundamentals_bulk(tickers)

    # 1) Prices stage (threaded)
    price_out: dict = {}

    def _price_one(sym: str) -> tuple:
        ok = True
        msg = ""
        saw_unauthorized = False

        if price_source == "tradingview":
            # Fetch from TradingView using new integration module
            from tradingview_integration import fetch_tradingview_prices
            ok, msg = fetch_tradingview_prices(sym, max_days=req.maxPriceDays)
        elif price_source == "eodhd":
            ok, msg = update_stock_data(client, sym, source="eodhd", max_days=req.maxPriceDays)
        else:
            # Fallback to Supabase check
            exists = check_local_cache(sym)
            ok = exists
            msg = f"OK ({price_source})" if exists else "Missing in Cloud"

        if (not ok) and ("401" in str(msg) or "Unauthorized" in str(msg)):
            saw_unauthorized = True

        return sym, ok, msg, saw_unauthorized

    if req.updatePrices:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(_price_one, sym): sym for sym in req.symbols}
            for fut in as_completed(future_map):
                sym, ok, msg, saw_unauthorized = fut.result()
                price_out[sym] = (ok, msg)
                if saw_unauthorized:
                    with unauthorized_lock:
                        unauthorized = True
    else:
        for sym in req.symbols:
            price_out[sym] = (True, "Skipped (price update disabled)")

    # 2) Fundamentals stage
    fund_out: dict = {}
    fund_ttl_seconds = int(os.getenv("FUND_TTL_SECONDS", str(60 * 60 * 24 * 30)))
    fund_error_ttl_seconds = int(os.getenv("FUND_ERROR_TTL_SECONDS", str(60 * 60 * 6)))
    now_ts = int(time.time())

    def _load_fund_cache(sym: str) -> Optional[tuple]:
        return None
        for cand_path in []: 
            if not os.path.exists(cand_path):
                continue
            try:
                with open(cand_path, "r") as f:
                    obj = json.load(f)

                if isinstance(obj, dict) and "data" in obj and "_meta" in obj:
                    data = obj.get("data") or {}
                    meta = obj.get("_meta") or {}
                elif isinstance(obj, dict):
                    data = obj
                    meta = {}
                else:
                    continue

                if not isinstance(data, dict) or not data:
                    # Might be an error cache
                    if isinstance(meta, dict) and meta.get("status") == "error":
                        fetched_at = meta.get("fetchedAt")
                        if isinstance(fetched_at, int) and (now_ts - fetched_at) <= fund_error_ttl_seconds:
                            return {}, {**meta, "servedFrom": "error_cached"}
                    continue

                if not _has_core_fund_metrics(data):
                    # Do not treat non-core-only payloads as fresh cache
                    continue

                if not isinstance(meta, dict) or not meta:
                    return data, {"servedFrom": "cache_legacy"}

                fetched_at = meta.get("fetchedAt")
                if isinstance(fetched_at, int) and (now_ts - fetched_at) <= fund_ttl_seconds:
                    return data, {**meta, "servedFrom": "cache_fresh"}

            except Exception:
                continue
        return None

    if req.updateFundamentals:
        # Determine desired provider per symbol
        per_symbol_provider: dict = {}
        tv_symbols: List[str] = []
        other_symbols: List[str] = []

        for sym in req.symbols:
            up = (sym or "").strip().upper()
            is_egx = up.endswith(".EGX") or up.endswith(".CC")

            if fund_source_cfg == "auto":
                # Do not force "mubasher" if auto. stock_ai.get_company_fundamentals handles the order (Mubasher -> TV).
                # provider = "mubasher" if is_egx else "tradingview"
                provider = "auto"
            else:
                provider = fund_source_cfg

            per_symbol_provider[sym] = provider
            # If auto (or tradingview), add to TV list for bulk fetch attempt
            if provider == "tradingview" or provider == "auto":
                tv_symbols.append(sym)
            else:
                other_symbols.append(sym)

        # 2a) TradingView bulk
        if tv_symbols:
            tv_to_fetch: List[str] = []
            for sym in tv_symbols:
                cached = _load_fund_cache(sym)
                if cached is not None:
                    fund_out[sym] = cached
                else:
                    tv_to_fetch.append(sym)

            if tv_to_fetch:
                bulk = _bulk_tradingview_fundamentals(tv_to_fetch)
                for sym in tv_to_fetch:
                    if sym in bulk:
                        fund_out[sym] = bulk[sym]

        # 2b) Per-symbol fundamentals for the rest + fallback for missing TradingView
        remaining: List[str] = [s for s in other_symbols if s not in fund_out]
        remaining += [s for s in tv_symbols if s not in fund_out]

        def _fund_one(sym: str) -> tuple:
            provider = per_symbol_provider.get(sym, fund_source_cfg)
            data, meta = get_company_fundamentals(sym, return_meta=True, source=provider)
            return sym, (data if isinstance(data, dict) else {}), (meta if isinstance(meta, dict) else {})

        if remaining:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_map = {ex.submit(_fund_one, sym): sym for sym in remaining}
                for fut in as_completed(future_map):
                    sym, data, meta = fut.result()
                    fund_out[sym] = (data, meta)
    else:
        for sym in req.symbols:
            fund_out[sym] = ({}, {"servedFrom": "skipped"})


    # 3) Merge
    for sym in req.symbols:
        price_ok, price_msg = price_out.get(sym, (False, "No price result"))

        fund_data, fund_meta = fund_out.get(sym, ({}, {}))
        fund_success = bool(fund_data)
        fund_status = fund_meta.get("servedFrom") if isinstance(fund_meta, dict) else None
        if not fund_status:
            fund_status = "ok" if fund_success else "unavailable"

        results.append(
            {
                "symbol": sym,
                "success": bool(price_ok),
                "message": f"{price_msg} | Fund: {fund_status}",
                "fund": {
                    "success": fund_success,
                    "source": fund_meta.get("source") if isinstance(fund_meta, dict) else None,
                    "servedFrom": fund_meta.get("servedFrom") if isinstance(fund_meta, dict) else None,
                    "data": _trim_fund_data(fund_data if isinstance(fund_data, dict) else {}),
                    "meta": fund_meta,
                },
            }
        )

    if unauthorized and price_source == "eodhd":
        return JSONResponse(
            status_code=401,
            content={
                "detail": "EODHD unauthorized. Check EODHD_API_KEY.",
                "results": results,
                "priceSource": price_source,
                "fundSource": fund_source_cfg,
            },
        )

    return {"results": results, "priceSource": price_source, "fundSource": fund_source_cfg, "debug_config_path": CONFIG_FILE}


@router.get("/db-inventory")
def get_db_inventory():
    """Retrieve accurate summary stats for all exchanges in Supabase using unified logic."""
    from api.stock_ai import get_supabase_inventory
    try:
        inventory = get_supabase_inventory()
        # Admin expects 'status' field for UI markers
        for item in inventory:
            item['status'] = "healthy" if item.get('price_count', 0) > 0 or item.get('fund_count', 0) > 0 else "empty"
            # Maintain backward compatibility for symbolCount if needed
            item['symbolCount'] = item.get('fund_count', 0)
        
        # Sort for admin UI consistency
        inventory.sort(key=lambda x: (x["priceCount"] == 0 and x["fundCount"] == 0, x["exchange"]))
        return inventory
    except Exception as e:
        print(f"Error fetching inventory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/db-symbols/{exchange}")
def get_db_symbols(exchange: str, mode: str = "prices"):
    """List all symbols for a specific exchange based on mode (prices or fundamentals)."""
    _init_supabase()
    if not supabase:
        return []
    
    try:
        symbols_info = {}
        if mode == "fundamentals":
            res = supabase.table("stock_fundamentals").select("symbol,updated_at,data").eq("exchange", exchange).execute()
            if res.data:
                for row in res.data:
                    s = row["symbol"]
                    d = row.get("data") or {}
                    symbols_info[s] = {
                        "symbol": s, "name": d.get("name", "N/A"), "sector": d.get("sector", "N/A"),
                        "last_sync": row["updated_at"], "last_price_date": None
                    }
        else:
            # Better logic: Fetch every unique symbol and its latest date for this exchange
            # We use an RPC for speed OR a more targeted query
            res = supabase.rpc("get_exchange_symbols_prices", {"p_exchange": exchange}).execute()
            if res.data:
                for row in res.data:
                    s = row["symbol"]
                    symbols_info[s] = {
                        "symbol": s, 
                        "name": "N/A", 
                        "sector": "N/A", 
                        "last_sync": None, 
                        "last_price_date": row["last_date"],
                        "row_count": row.get("count", 0)
                    }
        
        missing_meta = [s for s, info in symbols_info.items() if info["name"] == "N/A"]
        if missing_meta:
            # Chunking to avoid Supabase 500-symbol .in_ limit
            for chunk in _chunks(missing_meta, 500):
                res_meta = supabase.table("stock_fundamentals").select("symbol,data").eq("exchange", exchange).in_("symbol", chunk).execute()
                if res_meta.data:
                    for row in res_meta.data:
                        s = row["symbol"]
                        d = row.get("data") or {}
                        if s in symbols_info:
                            symbols_info[s]["name"] = d.get("name", d.get("Name", "N/A"))
                            symbols_info[s]["sector"] = d.get("sector", d.get("Sector", "N/A"))

        if mode == "prices":
            all_syms = list(symbols_info.keys())
            for chunk in _chunks(all_syms, 500):
                res_meta = supabase.table("stock_fundamentals").select("symbol,updated_at").eq("exchange", exchange).in_("symbol", chunk).execute()
                if res_meta.data:
                    for row in res_meta.data:
                        s = row["symbol"]
                        if s in symbols_info: 
                            symbols_info[s]["last_sync"] = row["updated_at"]

        return sorted(list(symbols_info.values()), key=lambda x: x["symbol"])
    except Exception as e:
        print(f"Error fetching symbols for {exchange}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export-prices/{exchange}")
def export_prices_csv(exchange: str, symbol: Optional[str] = None):
    """Export historical prices for an exchange or symbol as CSV."""
    _init_supabase()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not initialized")
    
    try:
        query = supabase.table("stock_prices").select("*").eq("exchange", exchange)
        if symbol:
            query = query.eq("symbol", symbol)
        
        # Limit to avoid huge memory usage, though for one exchange it's usually fine
        res = query.order("date", desc=True).limit(50000).execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail="No price data found")
            
        import pandas as pd
        df = pd.DataFrame(res.data)
        
        # Clean up for CSV
        if 'id' in df.columns: df = df.drop(columns=['id'])
        
        csv_data = df.to_csv(index=False)
        filename = f"{exchange}_{symbol or 'all'}_prices.csv"
        
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        print(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export-fundamentals/{exchange}")
def export_fundamentals_csv(exchange: str):
    """Export fundamentals for an exchange as CSV."""
    _init_supabase()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not initialized")
    
    try:
        res = supabase.table("stock_fundamentals").select("symbol, data, updated_at").eq("exchange", exchange).execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail="No fundamental data found")
            
        rows = []
        for item in res.data:
            data = item.get("data") or {}
            row = {
                "symbol": item["symbol"],
                "updated_at": item["updated_at"],
                **data
            }
            rows.append(row)
            
        import pandas as pd
        df = pd.DataFrame(rows)
        csv_data = df.to_csv(index=False)
        filename = f"{exchange}_fundamentals.csv"
        
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        print(f"Export fundamentals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync-history")
def get_sync_history(limit: int = 50):
    _init_supabase()
    if not supabase:
        return []
    try:
        res = supabase.table("data_sync_logs").select("*").order("started_at", desc=True).limit(limit).execute()
        return res.data
    except Exception as e:
        print(f"Error fetching sync history: {e}")
        return []

@router.get("/recent-fundamentals")
def get_recent_fundamentals(limit: int = 15):
    """Fetch the most recently updated fundamental records."""
    _init_supabase()
    if not supabase:
        return []
    try:
        res = supabase.table("stock_fundamentals").select("*").order("updated_at", desc=True).limit(limit).execute()
        return res.data
    except Exception as e:
        print(f"Error fetching recent fundamentals: {e}")
        return []
@router.post("/clear-prices")
def clear_prices():
    from api.stock_ai import clear_supabase_stock_prices
    ok, msg = clear_supabase_stock_prices()
    if not ok:
        raise HTTPException(status_code=500, detail=msg)
    return {"status": "success", "message": msg}

class RecalculateTechRequest(BaseModel):
    symbols: List[str]
    exchange: Optional[str] = None

@router.post("/recalculate-indicators")
def recalculate_indicators(req: RecalculateTechRequest, background_tasks: BackgroundTasks):
    """
    Recalculate technical indicators for the given symbols and save them to Supabase.
    Runs in background as it can be slow.
    """
    _reload_env()
    _init_supabase()
    api_key = os.getenv("EODHD_API_KEY")
    client = APIClient(api_key) if api_key else None
    
    symbols = list(req.symbols)
    if not symbols and req.exchange:
        # Fetch all symbols from stock_prices for this exchange using reliable RPC
        try:
            res = supabase.rpc("get_exchange_symbols_prices", {"p_exchange": req.exchange}).execute()
            if res.data:
                # RPC returns list of dicts with 'symbol' key
                symbols = [r["symbol"] for r in res.data]
                print(f"Recalculating ALL {len(symbols)} symbols for {req.exchange}")
            else:
                print(f"No symbols found via RPC for exchange {req.exchange}")
                return {"status": "error", "message": f"No symbols found for exchange {req.exchange}"}
        except Exception as e:
            print(f"Failed to fetch symbols for exchange {req.exchange} via RPC: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _worker(symbols_to_process: List[str], exchange_hint: Optional[str]):
        # Removed local symbols resolution as it's now handled in the main function

        for symbol in symbols_to_process:
            try:
                # 1. Infer exchange if not provided
                from api.stock_ai import _infer_symbol_exchange
                s, e = _infer_symbol_exchange(symbol, exchange_hint=req.exchange)
                full_symbol = f"{s}.{e}"
                
                # 2. Get historical data (force local if we already have it, otherwise fetch)
                # We need enough history for SMA_200 (at least 200 days)
                df = get_stock_data_eodhd(client, full_symbol, from_date="2023-01-01", exchange=e)
                
                if df is None or df.empty:
                    print(f"No data for {full_symbol} during recalculation")
                    continue
                
                # 3. Add technical indicators
                df_tech = add_technical_indicators(df)
                if df_tech.empty:
                    continue
                
                # 4. Extract latest row and sync
                last_row = df_tech.iloc[-1]
                date_str = df_tech.index[-1].strftime('%Y-%m-%d')
                
                # Daily change calculation
                close = float(last_row.get("Close", 0))
                prev_close = float(df_tech.iloc[-2].get("Close", close)) if len(df_tech) > 1 else close
                change_p = ((close - prev_close) / prev_close * 100) if prev_close != 0 else 0
                
                indicators = last_row.to_dict()
                indicators["change_p"] = change_p
                
                ok, msg = upsert_technical_indicators(
                    symbol=s,
                    exchange=e,
                    date=date_str,
                    close=close,
                    volume=int(last_row.get("Volume", 0)),
                    indicators=indicators
                )
                
                print(f"Recalculated {full_symbol}: {ok} - {msg}")
                
            except Exception as ex:
                print(f"Error recalculating {symbol}: {ex}")
                continue
                
    background_tasks.add_task(_worker, symbols, req.exchange)
    return {"status": "success", "message": f"Recalculation started for {len(symbols)} symbols in background"}

class TrainTriggerRequest(BaseModel):
    exchange: str
    useEarlyStopping: bool = True
    nEstimators: Optional[int] = None
    modelName: Optional[str] = None
    featurePreset: Optional[str] = None  # "core" | "extended" | "max"
    trainingStrategy: Optional[str] = None  # "golden" | "grid_small" | "random"
    randomSearchIter: Optional[int] = None
    maxFeatures: Optional[int] = None

@router.post("/train/trigger")
async def trigger_training(req: TrainTriggerRequest):
    _reload_env()
    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        # Fallback to checking if it's a local request, maybe we warn user?
        # For now, just raise error but maybe we can suggest local training
        raise HTTPException(status_code=500, detail="GITHUB_PAT not set. Configure it or use Local Training.")
    
    # Repository details
    owner = "weeasd57"
    repo = "stokscanai"
    workflow_id = "ai-training.yml"
    
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"
    headers = {
        "Authorization": f"token {github_pat}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "ref": "main", # Or the current branch
        "inputs": {
            "exchange": req.exchange
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 204:
            return {"status": "success", "message": f"Training triggered for {req.exchange}"}
        else:
            print(f"GitHub API Error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Failed to trigger GitHub Action: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train/local")
async def trigger_local_training(req: TrainTriggerRequest, background_tasks: BackgroundTasks):
    try:
        from train_exchange_model import train_model
    except ImportError as e:
        print(f"Import Error: {e}")
        if "lightgbm" in str(e):
             raise HTTPException(status_code=500, detail="LightGBM not installed. Server restarting... please try again in a moment.")
        raise HTTPException(status_code=500, detail=f"Failed to import training module: {e}")
    
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
         raise HTTPException(status_code=500, detail="Supabase credentials not configured")

    with _local_training_lock:
        LOCAL_TRAINING_STATE["running"] = True
        LOCAL_TRAINING_STATE["exchange"] = req.exchange
        LOCAL_TRAINING_STATE["started_at"] = datetime.utcnow().isoformat()
        LOCAL_TRAINING_STATE["completed_at"] = None
        LOCAL_TRAINING_STATE["error"] = None
        LOCAL_TRAINING_STATE["last_message"] = f"Starting local training for {req.exchange}"

    def _train_worker(
        ex,
        u,
        k,
        use_early_stopping: bool,
        n_estimators: Optional[int],
        model_name: Optional[str],
        training_strategy: Optional[str],
        random_search_iter: Optional[int],
        max_features: Optional[int],
    ):
        try:
            print(f"Starting local training for {ex}")

            def _progress_cb(msg) -> None:
                # Keep the admin UI responsive by updating last_message frequently.
                # This is best-effort and should never fail the training.
                try:
                    with _local_training_lock:
                        if isinstance(msg, dict):
                            message = msg.get("message")
                            if message is not None:
                                LOCAL_TRAINING_STATE["last_message"] = str(message)
                            phase = msg.get("phase")
                            if phase is not None:
                                LOCAL_TRAINING_STATE["phase"] = str(phase)
                            stats = msg.get("stats")
                            if isinstance(stats, dict):
                                LOCAL_TRAINING_STATE["stats"] = stats
                        else:
                            LOCAL_TRAINING_STATE["last_message"] = str(msg)
                        LOCAL_TRAINING_STATE["version"] = int(LOCAL_TRAINING_STATE.get("version") or 0) + 1
                        LOCAL_TRAINING_STATE["last_update"] = datetime.utcnow().isoformat()
                except Exception:
                    pass

            train_model(
                ex,
                u,
                k,
                use_early_stopping=use_early_stopping,
                n_estimators=n_estimators,
                model_name=model_name,
                upload_to_cloud=False,
                feature_preset=(req.featurePreset or "extended"),
                training_strategy=(training_strategy or "golden"),
                random_search_iter=random_search_iter,
                max_features=max_features,
                progress_cb=_progress_cb,
            )
            print(f"Local training for {ex} completed")
            with _local_training_lock:
                LOCAL_TRAINING_STATE["running"] = False
                LOCAL_TRAINING_STATE["completed_at"] = datetime.utcnow().isoformat()
                LOCAL_TRAINING_STATE["error"] = None
                LOCAL_TRAINING_STATE["last_message"] = f"Local training for {ex} completed"
                LOCAL_TRAINING_STATE["version"] = int(LOCAL_TRAINING_STATE.get("version") or 0) + 1
                LOCAL_TRAINING_STATE["last_update"] = datetime.utcnow().isoformat()
        except Exception as e:
            print(f"Local training failed: {e}")
            with _local_training_lock:
                LOCAL_TRAINING_STATE["running"] = False
                LOCAL_TRAINING_STATE["completed_at"] = datetime.utcnow().isoformat()
                LOCAL_TRAINING_STATE["error"] = str(e)
                LOCAL_TRAINING_STATE["last_message"] = f"Local training failed: {e}"
                LOCAL_TRAINING_STATE["version"] = int(LOCAL_TRAINING_STATE.get("version") or 0) + 1
                LOCAL_TRAINING_STATE["last_update"] = datetime.utcnow().isoformat()

    background_tasks.add_task(
        _train_worker,
        req.exchange,
        url,
        key,
        req.useEarlyStopping,
        req.nEstimators,
        req.modelName,
        req.trainingStrategy,
        req.randomSearchIter,
        req.maxFeatures,
    )
    return {"status": "success", "message": f"Local training started for {req.exchange}. Check server logs for progress."}


@router.get("/train/summary")
async def get_last_training_summary():
    """Return last training summary JSON written by train_exchange_model.py, if present."""
    try:
        api_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        summary_path = os.path.join(api_dir, "training_summary.json")
        if not os.path.exists(summary_path):
            return {"status": "empty"}
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "ok", "summary": data}
    except Exception as e:
        print(f"Failed to read training summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to read training summary")


@router.get("/train/status")
async def get_local_training_status():
    with _local_training_lock:
        return dict(LOCAL_TRAINING_STATE)


@router.get("/train/stream")
async def stream_local_training_status():
    def _event_stream():
        while True:
            with _local_training_lock:
                payload = dict(LOCAL_TRAINING_STATE)
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@router.get("/train/models")
async def list_models():
    """Expose local model artifacts for the admin UI using the same format as /admin/models/list.

    This avoids hitting Supabase storage and keeps the admin panel focused on server-side .pkl files.
    """
    return list_local_models()

@router.get("/train/download/{filename}")
async def get_model_download_url(filename: str):
    _init_supabase()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not initialized")
    
    try:
        # Create a signed URL for download (valid for 1 hour)
        res = supabase.storage.from_("ai-models").create_signed_url(filename, 3600)
        if 'signedURL' in res:
            return {"url": res['signedURL']}
        elif 'signedUrl' in res:
            return {"url": res['signedUrl']}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate signed URL")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
@router.post("/sync/trigger")
async def trigger_smart_sync(req: SmartSyncRequest):
    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        raise HTTPException(status_code=500, detail="GITHUB_PAT not configured")
    
    owner = "weeasd57"
    repo = "stokscanai"
    workflow_id = "data-sync.yml"
    
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"
    headers = {
        "Authorization": f"token {github_pat}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "ref": "main",
        "inputs": {
            "exchange": req.exchange,
            "days": str(req.days),
            "update_prices": "true" if req.updatePrices else "false",
            "update_funds": "true" if req.updateFunds else "false",
            "unified_dates": "true" if req.unified else "false"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 204:
            return {"status": "success", "message": f"Smart Sync triggered for {req.exchange}"}
        else:
            raise HTTPException(status_code=response.status_code, detail=f"GH API Error: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/actions/schedule")
async def schedule_action(req: ScheduleDispatchRequest, background_tasks: BackgroundTasks):
    """Schedule a one-time GitHub Actions dispatch at a future datetime."""
    _reload_env()
    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        raise HTTPException(status_code=500, detail="GITHUB_PAT not configured")

    owner = "weeasd57"
    repo = "stokscanai"

    wf = req.workflow.strip().lower()
    if wf in ("ai-training", "ai_training", "training"):
        workflow_id = "ai-training.yml"
        inputs = {"exchange": req.exchange or "US"}
    elif wf in ("data-sync", "data_sync", "sync"):
        workflow_id = "data-sync.yml"
        inputs = {
            "exchange": req.exchange or "US",
            "days": str(req.days or 365),
            "update_prices": "true" if (req.updatePrices is None or req.updatePrices) else "false",
            "update_funds": "true" if (req.updateFunds or False) else "false",
            "unified_dates": "true" if (req.unified or False) else "false",
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid workflow. Use 'ai-training' or 'data-sync'.")

    try:
        # Parse the target datetime
        when_str = req.when
        target_dt = datetime.fromisoformat(when_str.replace("Z", "+00:00")) if when_str else datetime.utcnow()
        # If naive, treat as UTC
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=None)
            now_utc = datetime.utcnow()
            delay = max(0, (target_dt - now_utc).total_seconds())
        else:
            now_utc_ts = datetime.utcnow().timestamp()
            delay = max(0, target_dt.timestamp() - now_utc_ts)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid 'when' datetime format. Use ISO 8601.")

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"
    headers = {
        "Authorization": f"token {github_pat}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {"ref": "main", "inputs": inputs}

    def _delayed_dispatch(wait_s: float, u: str, h: dict, p: dict):
        try:
            if wait_s > 0:
                time.sleep(wait_s)
            resp = requests.post(u, headers=h, json=p)
            if resp.status_code != 204:
                print(f"Schedule dispatch failed: {resp.status_code} - {resp.text}")
        except Exception as ex:
            print(f"Error during scheduled dispatch: {ex}")

    background_tasks.add_task(_delayed_dispatch, delay, url, headers, payload)
    return {"status": "scheduled", "workflow": workflow_id, "delay_seconds": int(delay)}

@router.post("/sync/schedule")
async def update_sync_schedule(req: ScheduleRequest):
    # This is a placeholder for updating the GHA YAML.
    # In a real scenario, this would involve git commit/push to the repo.
    # For now, we'll log it and return success for UI verification.
    print(f"Schedule update requested: {req.cron} (Window: {req.startTime} - {req.endTime})")
    return {"status": "success", "message": "Schedule preference updated (Log recorded)."}
@router.post("/logos/download")
def trigger_logo_download(req: LogoDownloadRequest, background_tasks: BackgroundTasks):
    try:
        from api.download_logos import download_logos
        background_tasks.add_task(download_logos, exchange=req.exchange, country=req.country)
        return {"status": "success", "message": "Logo download started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cron/trigger")
async def trigger_cron(req: CronTriggerRequest, background_tasks: BackgroundTasks):
    _reload_env()
    secret = os.getenv("CRON_SECRET")
    if not secret or req.secret != secret:
        raise HTTPException(status_code=401, detail="Invalid cron secret")
    
    def _cron_worker(action, ex_target):
        try:
            from api.stock_ai import get_supabase_inventory, get_supabase_symbols
            
            exchanges = [ex_target] if ex_target else [i['exchange'] for i in get_supabase_inventory() if i['fund_count'] > 0]
            
            for ex in exchanges:
                print(f"CRON [{action}]: Processing {ex}")
                if action == "recalculate_technicals":
                    # We call the existing logic
                    from api.routers.admin import RecalculateTechRequest, recalculate_indicators
                    recalculate_indicators(RecalculateTechRequest(symbols=[], exchange=ex), background_tasks)
                elif action in ["update_prices", "update_funds"]:
                    syms = [s['symbol'] for s in get_supabase_symbols(exchange=ex)]
                    for chunk in _chunks(syms, 50):
                        from api.routers.admin import UpdateRequest, update_batch
                        update_batch(UpdateRequest(
                            symbols=chunk, 
                            updatePrices=(action == "update_prices"), 
                            updateFundamentals=(action == "update_funds")
                        ), background_tasks)
                        time.sleep(1)
        except Exception as e:
            print(f"CRON Error: {e}")

    background_tasks.add_task(_cron_worker, req.action, req.exchange)
    return {"status": "success", "message": f"Cron action {req.action} started for {req.exchange or 'all exchanges'}"}

@router.get("/models/list")
def list_local_models():
    """List local model files from api/models directory, with optional metadata.

    Attempts to read basic metadata (num_features, num_parameters) when possible,
    but never fails the listing if a model cannot be loaded.
    """
    try:
        with _local_training_lock:
            is_training = bool(LOCAL_TRAINING_STATE.get("running"))
    except Exception:
        is_training = False

    try:
        import pickle
    except Exception:
        pickle = None

    try:
        # 1. Determine the models directory path robustly for local and Vercel environments
        # Start with the path relative to this file
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        models_dir = os.path.join(base_dir, "models")
        
        # Fallback: check relative to current working directory (Vercel /var/task)
        if not os.path.exists(models_dir):
            models_dir = os.path.join(os.getcwd(), "api", "models")
            
        # 2. Check if the directory exists
        if not os.path.exists(models_dir):
            print(f"Warning: Models directory not found at {models_dir}")
            return {"models": [], "error": "Models directory not found on server"}

        models = []
        for filename in os.listdir(models_dir):
            if not filename.endswith(".pkl"):
                continue

            filepath = os.path.join(models_dir, filename)
            if not os.path.isfile(filepath):
                continue

            stat = os.stat(filepath)
            size_bytes = stat.st_size
            size_mb = round(size_bytes / 1024 / 1024, 2)

            info = {
                "name": filename,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": "pkl",
            }

            # Best-effort metadata extraction
            if pickle is not None and (not is_training):
                try:
                    with open(filepath, "rb") as f:
                        model = pickle.load(f)

                    # Lightweight artifact format (fast to load)
                    if isinstance(model, dict) and model.get("kind") == "lgbm_booster":
                        nf = model.get("num_features")
                        nt = model.get("num_trees")
                        ne = model.get("n_estimators")
                        if isinstance(nf, (int, float)):
                            info["num_features"] = int(nf)
                        # Prefer n_estimators (what the user configured) as Params; fall back to num_trees
                        if isinstance(ne, (int, float)):
                            info["num_parameters"] = int(ne)
                        elif isinstance(nt, (int, float)):
                            info["num_parameters"] = int(nt)
                        ts = model.get("trainingSamples")
                        if isinstance(ts, (int, float)):
                            info["trainingSamples"] = int(ts)
                        # Done; avoid deeper inspection
                        models.append(info)
                        continue

                    # sklearn-style models often expose n_features_in_
                    num_features = getattr(model, "n_features_in_", None)
                    if isinstance(num_features, (int, float)):
                        num_features = int(num_features)
                        info["num_features"] = num_features

                    # Estimate an interpretable "parameter" count. The goal here is not exact
                    # theoretical parameters, but a relative measure of model capacity so that
                    # models with different n_estimators show different sizes in the UI.
                    num_params = getattr(model, "n_parameters_", None)

                    if isinstance(num_params, (int, float)):
                        # Some estimators expose an explicit parameter count
                        info["num_parameters"] = int(num_params)
                    else:
                        # Special handling for tree ensembles such as LightGBM / RandomForest
                        cls_name = type(model).__name__.lower()
                        if "lgbm" in cls_name or "gradientboost" in cls_name or "forest" in cls_name:
                            # For tree ensembles, surface the effective number of trees so that
                            # changing n_estimators (and early stopping) is visible in the UI.
                            trees = getattr(model, "n_estimators_", None)
                            if not isinstance(trees, (int, float)):
                                trees = getattr(model, "n_estimators", None)

                            if isinstance(trees, (int, float)):
                                info["num_parameters"] = int(trees)
                        else:
                            # Linear-style models: fall back to coef_ size when present
                            coef = getattr(model, "coef_", None)
                            if hasattr(coef, "size"):
                                try:
                                    info["num_parameters"] = int(coef.size)
                                except Exception:
                                    pass
                            else:
                                # As a last resort, try feature_importances_ length for tree models.
                                # This is less informative (usually == num_features), so only used
                                # when nothing else is available.
                                import numpy as _np  # local import to avoid top-level dependency if missing
                                fi = getattr(model, "feature_importances_", None)
                                if fi is not None:
                                    try:
                                        arr = _np.asarray(fi)
                                        info["num_parameters"] = int(arr.size)
                                    except Exception:
                                        pass
                except Exception as meta_err:
                    # Metadata is optional; just log and continue
                    print(f"Warning: failed to read metadata for model {filename}: {meta_err}")
            
            # Attach per-model metadata from sidecar JSON when available
            try:
                meta_path = os.path.join(models_dir, f"{filename}.meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    if isinstance(meta, dict):
                        ts = meta.get("trainingSamples")
                        if isinstance(ts, (int, float)):
                            info["trainingSamples"] = int(ts)
                        # Optionally surface other fields like exchange or featurePreset if useful later
                        ex = meta.get("exchange")
                        if isinstance(ex, str):
                            info.setdefault("exchange", ex)
                        fp = meta.get("featurePreset")
                        if isinstance(fp, str):
                            info.setdefault("featurePreset", fp)
            except Exception as meta_err:
                print(f"Warning: failed to read per-model meta for {filename}: {meta_err}")

            models.append(info)

        return {"models": sorted(models, key=lambda x: x["modified_at"], reverse=True)}
    except Exception as e:
        print(f"Error listing local models: {e}")
        return {"models": []}
    # HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/info")
def get_model_info(model_name: str):
    """Get model info including parameters and features count."""
    try:
        if ".." in model_name or "/" in model_name or "\\" in model_name:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        if not model_name.endswith(".pkl"):
            raise HTTPException(status_code=400, detail="Only .pkl files supported")
        
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        filepath = os.path.join(models_dir, model_name)
        
        if not os.path.abspath(filepath).startswith(models_dir):
            raise HTTPException(status_code=400, detail="Invalid model path")
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Model not found")
        
        import pickle
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        # Lightweight artifact format
        if isinstance(model, dict) and model.get("kind") == "lgbm_booster":
            features = model.get("feature_names")
            if not isinstance(features, list):
                features = []
            nf = model.get("num_features")
            nt = model.get("num_trees")
            ne = model.get("n_estimators")
            info = {
                "name": model_name,
                "num_features": int(nf) if isinstance(nf, (int, float)) else len(features),
                "num_parameters": int(ne) if isinstance(ne, (int, float)) else int(nt) if isinstance(nt, (int, float)) else 0,
                "model_type": "lgbm_booster",
                "features": features,
            }

            # Optional metadata
            for k in ("exchange", "featurePreset", "trainingStrategy", "timestamp", "trainingSamples"):
                v = model.get(k)
                if v is not None:
                    info[k] = v

            return info
        
        # Extract model info
        info = {
            "name": model_name,
            "num_features": 0,
            "num_parameters": 0,
            "model_type": type(model).__name__,
            "features": []
        }
        
        # Get number of features
        if hasattr(model, 'n_features_in_'):
            info["num_features"] = int(model.n_features_in_)
        
        # Get feature names
        if hasattr(model, 'feature_names_in_'):
            info["features"] = list(model.feature_names_in_)
        
        # Get number of parameters/estimators
        if hasattr(model, 'n_estimators'):
            info["num_parameters"] = int(model.n_estimators)
        elif hasattr(model, 'n_components_'):
            info["num_parameters"] = int(model.n_components_)
        elif hasattr(model, 'get_params'):
            info["num_parameters"] = len(model.get_params())
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _delete_local_model(model_name: str):
    """Delete a local model file."""
    try:
        # Validate filename to prevent path traversal
        if ".." in model_name or "/" in model_name or "\\" in model_name:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        if not model_name.endswith(".pkl"):
            raise HTTPException(status_code=400, detail="Only .pkl files can be deleted")
        
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        filepath = os.path.join(models_dir, model_name)
        
        # Ensure the filepath is within models_dir
        if not os.path.abspath(filepath).startswith(models_dir):
            raise HTTPException(status_code=400, detail="Invalid model path")
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Model not found")
        
        os.remove(filepath)
        return {"status": "success", "message": f"Model {model_name} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_name}")
def delete_local_model(model_name: str):
    return _delete_local_model(model_name)

@router.delete("/train/models/{model_name}")
def delete_local_model_from_train(model_name: str):
    return _delete_local_model(model_name)

@router.post("/update-symbols-inventory")
async def update_symbols_inventory(background_tasks: BackgroundTasks, country: Optional[str] = None):
    """
    Fetch latest exchange list and symbols inventory from EODHD.
    Updates files in symbols_data.
    """
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="EODHD_API_KEY not set")

    def _inventory_worker():
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "symbols_data")
        os.makedirs(base_dir, exist_ok=True)
        
        def _cleanup_old_files(prefix: str):
            """Delete old files with the same prefix to prevent duplicates."""
            try:
                for f in os.listdir(base_dir):
                    if f.startswith(prefix) and f.endswith(".json") and "_all_symbols_" in f:
                        os.remove(os.path.join(base_dir, f))
                # Also cleanup summary if updating global
                if prefix in ["country_summary", "all_symbols_by_country"]:
                    for f in os.listdir(base_dir):
                        if f.startswith(prefix) and f.endswith(".json"):
                            os.remove(os.path.join(base_dir, f))
            except Exception as e:
                print(f"Cleanup error for {prefix}: {e}")

        try:
            # 1. Fetch Exchange List
            url = f"https://eodhd.com/api/exchanges-list/?api_token={api_key}&fmt=json"
            print(f"Fetching exchange list from EODHD...")
            import urllib.request
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            # Save raw exchange list (always overwrite or cleanup)
            ex_list_path = os.path.join(base_dir, "exchanges_list.json")
            with open(ex_list_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            # 2. Process into a country summary
            from api.symbols_local import load_country_summary
            country_summary = load_country_summary() or {}
            all_symbols = [] # This will be the new global list if full update
            
            is_full_update = country is None
            target_country = country.strip() if country else None
            
            # Major exchanges to fetch by default in full update
            major_exchanges = ["US", "EGX", "LSE", "TO", "V", "PA", "F", "MI"] if is_full_update else []

            for ex in data:
                c_name = ex.get("Country", "Unknown")
                ex_code = ex.get("Code", "")
                
                # Filter logic
                should_fetch = False
                if not is_full_update:
                    if target_country and c_name.lower() == target_country.lower():
                        should_fetch = True
                elif ex_code in major_exchanges:
                    should_fetch = True
                
                if should_fetch:
                     try:
                         sym_url = f"https://eodhd.com/api/exchange-symbol-list/{ex_code}?api_token={api_key}&fmt=json"
                         print(f"Fetching symbol list for {ex_code} ({c_name})...")
                         with urllib.request.urlopen(sym_url, timeout=60) as sresp:
                             syms = json.loads(sresp.read().decode("utf-8"))
                             
                             # Normalization: Map 'Code' to 'Symbol' for legacy compatibility
                             normalized_syms = []
                             for s in syms:
                                 n = {
                                     "Symbol": s.get("Code"),
                                     "Name": s.get("Name"),
                                     "Exchange": ex_code,
                                     "Country": c_name,
                                     "Type": s.get("Type"),
                                     "Currency": s.get("Currency"),
                                     "Isin": s.get("Isin")
                                 }
                                 normalized_syms.append(n)
                             
                             # Cleanup old version of this country's file
                             _cleanup_old_files(f"{c_name}_all_symbols")
                             
                             # Save individual country file
                             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                             ex_file = os.path.join(base_dir, f"{c_name}_all_symbols_{timestamp}.json")
                             with open(ex_file, "w", encoding="utf-8") as f:
                                 json.dump(normalized_syms, f, indent=2)
                                 
                             # Update counts in summary
                             if c_name not in country_summary:
                                 country_summary[c_name] = {"TotalSymbols": 0, "Exchanges": {}}
                             
                             ex_count = len(normalized_syms)
                             # Reset count for this specific exchange to avoid double counting if partial update
                             old_ex_count = country_summary[c_name]["Exchanges"].get(ex_code, 0)
                             country_summary[c_name]["Exchanges"][ex_code] = ex_count
                             country_summary[c_name]["TotalSymbols"] = country_summary[c_name]["TotalSymbols"] - old_ex_count + ex_count
                             
                             if is_full_update:
                                 all_symbols.extend(normalized_syms)
                                 
                     except Exception as e:
                         print(f"Error fetching symbols for {ex_code}: {e}")

            # 3. Save Summary Files (Only in full update or to persist summary changes)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            _cleanup_old_files("country_summary")
            summary_path = os.path.join(base_dir, f"country_summary_{timestamp}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(country_summary, f, indent=2)
            
            if is_full_update:
                _cleanup_old_files("all_symbols_by_country")
                all_syms_path = os.path.join(base_dir, f"all_symbols_by_country_{timestamp}.json")
                with open(all_syms_path, "w", encoding="utf-8") as f:
                    json.dump(all_symbols, f, indent=2)
                
            print("Symbol inventory update complete.")
            
        except Exception as e:
            print(f"Error updating symbol inventory: {e}")

    background_tasks.add_task(_inventory_worker)
    return {"status": "success", "message": f"Inventory update {'for ' + country if country else 'started'} in background"}
