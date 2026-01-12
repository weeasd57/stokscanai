import os
import json
from typing import List, Optional
from collections import defaultdict
import time
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from eodhd import APIClient
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from supabase import create_client, Client


from api.stock_ai import (
    get_stock_data, get_stock_data_eodhd, get_company_fundamentals,
    _init_supabase, supabase, update_stock_data,
    _finite_float,
)

router = APIRouter(prefix="/admin", tags=["admin"])

class UpdateRequest(BaseModel):
    symbols: List[str]
    country: Optional[str] = None
    updatePrices: bool = True
    updateFundamentals: bool = False
    maxPriceDays: int = 365

class SyncRequest(BaseModel):
    exchange: Optional[str] = None # If None, sync all
    force: bool = False # If True, re-upload even if no change (not used yet)



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

    def _chunks(items: List[str], size: int) -> List[List[str]]:
        if size <= 0:
            return [items]
        return [items[i : i + size] for i in range(0, len(items), size)]

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
                provider = "mubasher" if is_egx else "tradingview"
            else:
                provider = fund_source_cfg

            per_symbol_provider[sym] = provider
            if provider == "tradingview":
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


@router.get("/usage")
def get_usage():
    import requests
    _reload_env()
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="EODHD_API_KEY not set")

    try:
        url = f"https://eodhd.com/api/user?api_token={api_key}&fmt=json"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()

        # Structure typically:
        # { "apiRequests": 19, "dailyRateLimit": 20, ... }
        # Extra credits might be in different field or inferred.
        # But user specifically mentioned logic "100110 left".
        # Let's return the whole object or mapped fields.

        return {
            "used": data.get("apiRequests", 0),
            "limit": data.get("dailyRateLimit", 0),
            "extraLeft": data.get("paymentTokens", 0) # Assuming this maps to extra credits usually
        }
    except Exception as e:
        print(f"Error fetching usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))





@router.get("/db-inventory")
def get_db_inventory():
    """Retrieve summary stats for all exchanges currently in Supabase."""
    _init_supabase()
    if not supabase:
        return []
    
    try:
        # Get price stats
        # Supabase doesn't support GROUP BY via the client easily for aggregations.
        # We'll fetch the data and group in Python for simplicity, 
        # or use a raw query if we had one. Since we don't have thousands of exchanges, this is fine.
        
        # Actually, let's try a clever select to get some info.
        # For a small-ish number of rows, we can just get all unique symbol/exchange pairs.
        # But maybe better to just use a custom RPC if we want it perfect.
        # However, we'll try to get it via the client.
        
        # We'll get fundamentals as they are 1 row per symbol
        res_fund = supabase.table("stock_fundamentals").select("exchange,updated_at,data").execute()
        
        inventory = defaultdict(lambda: {"prices": 0, "fundamentals": 0, "last_update": None, "country": "N/A"})
        
        if res_fund.data:
            for row in res_fund.data:
                ex = row["exchange"] or "UNKNOWN"
                inventory[ex]["fundamentals"] += 1
                upd = row["updated_at"]
                
                # Fetch country from data if not already set for this exchange
                if inventory[ex]["country"] == "N/A":
                    d = row.get("data") or {}
                    country = d.get("country")
                    if country:
                        inventory[ex]["country"] = country

                if not inventory[ex]["last_update"] or (upd and upd > inventory[ex]["last_update"]):
                    inventory[ex]["last_update"] = upd

        # Also get price counts (rough)
        # We can't easily count distinct symbols per exchange via client. 
        # We'll just return the fundamental counts as the "symbol count" for now.
        
        out = []
        for ex, stats in inventory.items():
            out.append({
                "exchange": ex,
                "country": stats["country"],
                "symbolCount": stats["fundamentals"],
                "lastUpdate": stats["last_update"],
                "status": "healthy" if stats["fundamentals"] > 0 else "empty"
            })
            
        return sorted(out, key=lambda x: x["exchange"])
    except Exception as e:
        print(f"Error fetching inventory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/db-symbols/{exchange}")
def get_db_symbols(exchange: str):
    """List all symbols for a specific exchange and their status."""
    _init_supabase()
    if not supabase:
        return []
    
    try:
        # 1. Get fundamental records (provides names and last sync times)
        res_fund = supabase.table("stock_fundamentals").select("symbol,updated_at,data").eq("exchange", exchange).execute()
        
        # 2. Try to get latest price dates for these symbols
        # Since we can't easily group_by, we fetch recent price rows and pick the max per symbol
        # We fetch a larger batch to improve chances of hitting all symbols
        res_prices = supabase.table("stock_prices").select("symbol,date").eq("exchange", exchange).order("date", desc=True).limit(1000).execute()
        
        latest_dates = {}
        if res_prices.data:
            for row in res_prices.data:
                s = row["symbol"]
                d = row["date"]
                if s not in latest_dates or d > latest_dates[s]:
                    latest_dates[s] = d

        if not res_fund.data:
            return []
            
        out = []
        for row in res_fund.data:
            s_code = row["symbol"]
            d = row.get("data") or {}
            out.append({
                "symbol": s_code,
                "name": d.get("name", "N/A"),
                "last_sync": row["updated_at"],
                "last_price_date": latest_dates.get(s_code),
                "sector": d.get("sector", "N/A"),
            })

        return sorted(out, key=lambda x: x["symbol"])
    except Exception as e:
        print(f"Error fetching symbols for {exchange}: {e}")
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
