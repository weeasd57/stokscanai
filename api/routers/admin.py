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
    _init_supabase, supabase, update_stock_data, sync_local_to_supabase,
    _candidate_cache_paths,
    _candidate_fund_cache_paths,
    _preferred_fund_cache_path,
    _safe_mkdir,
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
ENV_API_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

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
    load_dotenv(ENV_API_FILE, override=True)

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
    cache_dir = os.getenv("CACHE_DIR", "data_cache")
    cfg = _load_config()
    selected = (source or cfg.get("fundSource") or "auto")
    data, meta = get_company_fundamentals(ticker, cache_dir=cache_dir, return_meta=True, source=selected)
    return {"ticker": ticker, "data": data, "meta": meta}

@router.post("/update_batch")
def update_batch(req: UpdateRequest, background_tasks: BackgroundTasks):
    _reload_env()
    api_key = os.getenv("EODHD_API_KEY")
    cache_dir = os.getenv("CACHE_DIR", "data_cache")
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
        return fetch_tradingview_fundamentals_bulk(tickers, cache_dir=cache_dir)

    # 1) Prices stage (threaded)
    price_out: dict = {}

    def _price_one(sym: str) -> tuple:
        ok = True
        msg = ""
        saw_unauthorized = False

        if price_source == "tradingview":
            # Fetch from TradingView using new integration module
            from tradingview_integration import fetch_tradingview_prices
            ok, msg = fetch_tradingview_prices(sym, cache_dir=cache_dir, max_days=req.maxPriceDays)
        elif price_source == "eodhd":
            ok, msg = update_stock_data(client, sym, source="eodhd", cache_dir=cache_dir, max_days=req.maxPriceDays)
        else:
            # Fallback to cache check for other sources
            cache_ok = False
            for p in _candidate_cache_paths(cache_dir, sym):
                if os.path.exists(p):
                    cache_ok = True
                    break
            ok = cache_ok
            msg = f"OK ({price_source})" if cache_ok else "Missing local cache"

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
        for cand_path in _candidate_fund_cache_paths(cache_dir, sym):
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
            data, meta = get_company_fundamentals(sym, cache_dir=cache_dir, return_meta=True, source=provider)
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




@router.post("/sync-data")
def sync_data(req: SyncRequest, background_tasks: BackgroundTasks):
    _reload_env()
    _init_supabase()
    # api_key = os.getenv("EODHD_API_KEY") # No longer strictly required for sync
    # if not api_key:
    #     raise HTTPException(status_code=500, detail="EODHD_API_KEY not set")

    # Start log
    log_entry = {
        "exchange": req.exchange or "ALL",
        "started_at": datetime.utcnow().isoformat(),
        "status": "running",
        "triggered_by": "admin"
    }
    log_id = None
    if supabase:
        try:
            res = supabase.table("data_sync_logs").insert(log_entry).execute()
            if res.data:
                log_id = res.data[0]['id']
        except Exception as e:
            print(f"Failed to create sync log: {e}")

    # Background Task for Sync (Local Upload Only)
    def _do_sync(log_id):
        from api.stock_ai import _resolve_cache_dir
        _init_supabase()
        
        env_cache = os.getenv("CACHE_DIR", "data_cache")
        cache_dir = _resolve_cache_dir(env_cache)
        print(f"DEBUG: Background sync started. LogId={log_id}, CacheDir={cache_dir}, Exchange={req.exchange or 'ALL'}")

        symbols_updated = 0
        prices_updated = 0
        
        # Collect all tasks first
        tasks = []
        
        from api.symbols_local import list_countries, load_symbols_for_country
        
        # Discover symbols from symbols_data defining JSONs
        discovery_countries = []
        if not req.exchange or req.exchange == "ALL":
            discovery_countries = list_countries()
        else:
            # Map exchange code to country name
            mapping = {"EGX": "Egypt", "US": "USA"}
            country = mapping.get(req.exchange.upper())
            if country:
                discovery_countries = [country]
            else:
                discovery_countries = [req.exchange] # Try as is
            
        print(f"DEBUG: Sync - Scanning countries: {discovery_countries}")
        
        total_files_scanned = 0
        for country in discovery_countries:
            try:
                symbols_data = load_symbols_for_country(country)
                ex_tickers = set()
                
                # Heuristic for exchange code from country
                ex_mapping = {"Egypt": "EGX", "USA": "US"}
                ex_code = ex_mapping.get(country, "US") # Default to US

                for row in symbols_data:
                    s = str(row.get("Code", row.get("Symbol", "")))
                    if s:
                        ex_tickers.add(s)
                
                print(f"DEBUG: Sync - Discovered {len(ex_tickers)} tickers for {country} (Mapped Ex: {ex_code})")
                for ticker in ex_tickers:
                    # Filter if specific exchange was requested and this ticker doesn't match
                    if req.exchange and req.exchange != "ALL" and req.exchange.upper() != ex_code:
                        continue
                    tasks.append((ticker, ex_code))
                    total_files_scanned += 1
            except Exception as e:
                print(f"DEBUG: Sync - Error loading symbols for {country}: {e}")
                continue
        
        if not tasks:
            print("DEBUG: Background sync finished. No tasks found.")
            if log_id and supabase:
                supabase.table("data_sync_logs").update({
                    "completed_at": datetime.utcnow().isoformat(),
                    "status": "success",
                    "symbols_updated": 0,
                    "prices_updated": 0,
                    "notes": "No local cache files found to sync."
                }).eq("id", log_id).execute()
            return

        print(f"DEBUG: Background sync executing {len(tasks)} tasks with 10 workers")
        
        # Run in threads
        import concurrent.futures
        max_workers = 10 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            def _upload_task(s, e):
                from api.stock_ai import update_stock_data, get_company_fundamentals, APIClient
                
                # EODHD Client for updates
                token = os.getenv("EODHD_API_KEY")
                api_client = APIClient(token) if token else None

                price_updated = False
                fund_updated = False
                
                # 1. Update/Sync Prices (In-memory)
                if api_client:
                    ok, _ = update_stock_data(api_client, f"{s}.{e}")
                    if ok: price_updated = True
                
                # 2. Update/Sync Fundamentals (In-memory)
                # get_company_fundamentals now calls sync_data_to_supabase internally
                get_company_fundamentals(f"{s}.{e}", source="auto")
                fund_updated = True # Assume success or it logged error
                    
                if price_updated or fund_updated:
                    return True, "Synced to Cloud"
                return False, "Failed to update from API"

            future_to_sym = {
                executor.submit(_upload_task, sym, ex): sym 
                for (sym, ex) in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    ok, msg = future.result()
                    if ok:
                        symbols_updated += 1
                        prices_updated += 1 # We'll use this as total count for now
                        if symbols_updated % 50 == 0:
                            print(f"DEBUG: Sync progress: {symbols_updated}/{len(tasks)} symbols processed...")
                    else:
                        if "Nothing" not in msg:
                            print(f"DEBUG: Sync failed for {sym}: {msg}")
                except Exception as e:
                    print(f"DEBUG: Critical error syncing {sym}: {e}")

        print(f"DEBUG: Background sync completed. Total success: {symbols_updated}/{len(tasks)}")

        # Update log
        if log_id and supabase:
            try:
                supabase.table("data_sync_logs").update({
                    "completed_at": datetime.utcnow().isoformat(),
                    "status": "success",
                    "symbols_updated": symbols_updated,
                    "prices_updated": prices_updated
                }).eq("id", log_id).execute()
            except Exception as e:
                print(f"DEBUG: Error updating final log: {e}")

    background_tasks.add_task(_do_sync, log_id)
    return {"status": "started", "log_id": log_id}

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
        res_fund = supabase.table("stock_fundamentals").select("exchange,updated_at").execute()
        
        inventory = defaultdict(lambda: {"prices": 0, "fundamentals": 0, "last_update": None})
        
        if res_fund.data:
            for row in res_fund.data:
                ex = row["exchange"] or "UNKNOWN"
                inventory[ex]["fundamentals"] += 1
                upd = row["updated_at"]
                if not inventory[ex]["last_update"] or (upd and upd > inventory[ex]["last_update"]):
                    inventory[ex]["last_update"] = upd

        # Also get price counts (rough)
        # We can't easily count distinct symbols per exchange via client. 
        # We'll just return the fundamental counts as the "symbol count" for now.
        
        out = []
        for ex, stats in inventory.items():
            out.append({
                "exchange": ex,
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
        # Get fundamentals (1 row per symbol)
        res = supabase.table("stock_fundamentals").select("symbol,updated_at").eq("exchange", exchange).execute()
        
        if not res.data:
            return []
            
        return sorted(res.data, key=lambda x: x["symbol"])
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
