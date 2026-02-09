# RESTART_DEBUG: 2
import os
import sys
import json
import datetime as dt
import urllib.request
import pandas as pd
import numpy as np
import warnings

# Suppress specific FutureWarnings from libraries like 'ta'
warnings.filterwarnings("ignore", category=FutureWarning)

# from dotenv import load_dotenv

from dotenv import load_dotenv
# Explicitly load .env from the root directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))
load_dotenv(os.path.join(base_dir, "web", ".env.local"), override=True)

print(f"DEBUG: EODHD_API_KEY loaded: {'Yes' if os.getenv('EODHD_API_KEY') else 'No'}")

from fastapi import FastAPI, HTTPException, Request, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, Optional, List, Literal

from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

import yfinance as yf

print("Main: Importing stock_ai...")
from api.stock_ai import run_pipeline
print("Main: Importing symbols_local...")
from api.symbols_local import list_countries, search_symbols
print("Main: Importing routers...")
from api.routers import scan_ai, scan_ai_fast, scan_tech, admin, alpaca
print("Main: Imports done.")

load_dotenv()

# Debug: Print loaded environment variables
print(f"DEBUG: NEXT_PUBLIC_SUPABASE_URL loaded: {'Yes' if os.getenv('NEXT_PUBLIC_SUPABASE_URL') else 'No'}")
print(f"DEBUG: SUPABASE_SERVICE_ROLE_KEY loaded: {'Yes' if os.getenv('SUPABASE_SERVICE_ROLE_KEY') else 'No'}")

app = FastAPI(title="Artoro API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    # Initialize Supabase at startup
    from api.stock_ai import _init_supabase
    _init_supabase()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    print(f"Unhandled exception for {request.method} {request.url.path}: {exc}")
    import traceback
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {str(exc)}"})

app.include_router(scan_ai.router)
app.include_router(scan_ai_fast.router)
app.include_router(scan_tech.router)
app.include_router(admin.router)
app.include_router(alpaca.router)
from api.routers import bot
app.include_router(bot.router, prefix="/ai_bot")
app.include_router(bot.router, prefix="/bot") # Compatibility Alias

CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "admin_config.json"))


def _load_admin_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"source": "eodhd"}


def _normalize_yahoo_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.endswith(".US"):
        return t.replace(".US", "")
    if t.endswith(".EGX"):
        return t.replace(".EGX", ".CA")
    return t


def _fetch_price_yahoo(ticker: str) -> float:
    yf_ticker = _normalize_yahoo_ticker(ticker)
    t = yf.Ticker(yf_ticker)

    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            last = fi.get("last_price")
            if last is not None:
                return float(last)
    except Exception:
        pass

    try:
        hist = t.history(period="1d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass

    raise ValueError("Yahoo price unavailable")


def _fetch_price_eodhd(ticker: str, api_key: str) -> float:
    url = f"https://eodhd.com/api/real-time/{ticker}?api_token={api_key}&fmt=json"
    with urllib.request.urlopen(url, timeout=20) as resp:
        raw = resp.read().decode("utf-8")
    payload = json.loads(raw)

    if not isinstance(payload, dict):
        raise ValueError("Invalid EODHD response")

    for k in ("close", "price", "last", "last_close"):
        v = payload.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue

    raise ValueError("EODHD price unavailable")

web_origin = os.getenv("WEB_ORIGIN", "*")
allow_origins = [web_origin] if web_origin != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "PATCH", "PUT", "OPTIONS"],
    allow_headers=["*"]
)


class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=24, pattern=r"^[A-Za-z0-9.\-]{1,24}$")
    exchange: Optional[str] = Field(default=None)
    from_date: str = Field(default="2020-01-01")
    to_date: Optional[str] = Field(default=None)
    include_fundamentals: bool = Field(default=True)
    rf_preset: Optional[str] = Field(default=None)
    rf_params: Optional[Dict[str, Any]] = Field(default=None)
    model_name: Optional[str] = Field(default=None)
    force_local: bool = Field(default=False)
    target_pct: float = Field(default=0.15)
    stop_loss_pct: float = Field(default=0.05)
    look_forward_days: int = Field(default=20)
    buy_threshold: float = Field(default=0.45)
    use_volatility_label: bool = Field(default=False)


class EvaluatePositionIn(BaseModel):
    id: str
    symbol: str
    entry_price: Optional[float] = None
    entry_at: Optional[str] = Field(default=None, description="ISO timestamp")
    added_at: Optional[str] = Field(default=None, description="ISO timestamp")
    target_price: Optional[float] = None
    stop_price: Optional[float] = None


class EvaluatePositionsRequest(BaseModel):
    positions: List[EvaluatePositionIn]


class EvaluatePositionOut(BaseModel):
    id: str
    symbol: str
    status: Literal["open", "hit_target", "hit_stop"]
    as_of: Optional[str] = None
    price: Optional[float] = None
    change_pct: Optional[float] = None
    reason: Optional[str] = None


def _parse_iso_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        d = dt.datetime.fromisoformat(v)
        return d.date().isoformat()
    except Exception:
        try:
            return dt.date.fromisoformat(v[:10]).isoformat()
        except Exception:
            return None


def _fetch_eod_history_eodhd(ticker: str, api_key: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
    url = (
        f"https://eodhd.com/api/eod/{ticker}"
        f"?api_token={api_key}&fmt=json&period=d&order=a&from={from_date}&to={to_date}"
    )
    with urllib.request.urlopen(url, timeout=25) as resp:
        raw = resp.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("Invalid EODHD EOD response")
    return payload


@app.post("/positions/evaluate_open_history", response_model=List[EvaluatePositionOut])
def evaluate_open_positions_history(req: EvaluatePositionsRequest):
    from tradingview_integration import fetch_tradingview_prices
    
    api_key = os.getenv("EODHD_API_KEY")
    today = dt.datetime.utcnow().date().isoformat()
    out: List[EvaluatePositionOut] = []

    for p in req.positions:
        start_date = _parse_iso_date(p.entry_at) or _parse_iso_date(p.added_at)
        if not start_date:
            out.append(EvaluatePositionOut(id=p.id, symbol=p.symbol, status="open", reason="missing_start_date"))
            continue

        if p.target_price is None and p.stop_price is None:
            out.append(EvaluatePositionOut(id=p.id, symbol=p.symbol, status="open", reason="missing_target_stop"))
            continue

        symbol = p.symbol.strip().upper()
        # Standardize symbol/exchange inference
        from api.stock_ai import _infer_symbol_exchange, get_stock_data_eodhd, _finite_float
        from eodhd import APIClient

        s, e = _infer_symbol_exchange(symbol)
        full_symbol = f"{s}.{e}"
        
        # Try to update from TradingView first (free)
        try:
            fetch_tradingview_prices(full_symbol, max_days=500)
        except Exception as ex:
            print(f"TradingView update failed for {full_symbol}: {ex}")
        
        # Use centralized get_stock_data_eodhd which handles Supabase -> Local -> API
        df_loaded = None
        try:
            api_client = APIClient(api_key) if api_key else None
            df_loaded = get_stock_data_eodhd(
                api=api_client,
                ticker=full_symbol,
                from_date=start_date,
                exchange=e
            )
        except Exception as ex:
            print(f"Data fetch error for {full_symbol}: {ex}")
            # If no data and we have no API key, it's a real failure
            if not api_key:
                out.append(EvaluatePositionOut(id=p.id, symbol=p.symbol, status="open", reason="no_data_source"))
                continue
            out.append(EvaluatePositionOut(id=p.id, symbol=p.symbol, status="open", reason=f"fetch_error:{ex}"))
            continue

        if df_loaded is None or df_loaded.empty:
            out.append(EvaluatePositionOut(id=p.id, symbol=p.symbol, status="open", reason="no_data"))
            continue
        
        # We have the dataframe, ensure it's sorted and has a proper index
        if not isinstance(df_loaded.index, pd.DatetimeIndex):
            df_loaded.index = pd.to_datetime(df_loaded.index)
        
        df_filtered = df_loaded[df_loaded.index >= pd.to_datetime(start_date)].sort_index()
        
        if df_filtered.empty:
            out.append(EvaluatePositionOut(id=p.id, symbol=p.symbol, status="open", reason="no_data_in_range"))
            continue
        
        # Evaluate hits
        hit: Optional[EvaluatePositionOut] = None
        for timestamp, row in df_filtered.iterrows():
            try:
                # timestamp is a pd.Timestamp here
                d = timestamp.strftime('%Y-%m-%d')
                # EODHD/Supabase use lowercase column names
                high_v = _finite_float(row.get('high', row.get('High')))
                low_v = _finite_float(row.get('low', row.get('Low')))
            except Exception:
                continue
            
            hit_target = bool(p.target_price is not None and high_v is not None and high_v >= float(p.target_price))
            hit_stop = bool(p.stop_price is not None and low_v is not None and low_v <= float(p.stop_price))
            
            if hit_target and hit_stop:
                hit = EvaluatePositionOut(
                    id=p.id, symbol=p.symbol, status="hit_stop",
                    as_of=d, price=float(p.stop_price) if p.stop_price else None,
                    reason="both_crossed_same_day"
                )
                break
            
            if hit_stop:
                hit = EvaluatePositionOut(
                    id=p.id, symbol=p.symbol, status="hit_stop",
                    as_of=d, price=float(p.stop_price) if p.stop_price else None,
                    reason="low<=stop"
                )
                break
            
            if hit_target:
                hit = EvaluatePositionOut(
                    id=p.id, symbol=p.symbol, status="hit_target",
                    as_of=d, price=float(p.target_price) if p.target_price else None,
                    reason="high>=target"
                )
                break
        
        if hit is None:
            # Always return the latest price/date even if no hit
            last_idx = df_filtered.index[-1]
            last_row = df_filtered.iloc[-1]
            last_price = float(last_row.get('close', last_row.get('Close')))
            last_date = last_idx.strftime('%Y-%m-%d')
            
            cp = None
            if p.entry_price and last_price:
                cp = ((last_price - float(p.entry_price)) / float(p.entry_price)) * 100

            out.append(EvaluatePositionOut(
                id=p.id, symbol=p.symbol, status="open",
                as_of=last_date, price=last_price,
                change_pct=cp,
                reason="no_hit"
            ))
        else:
            # For hits, calculate change_pct based on the hit price
            if p.entry_price and hit.price:
                hit.change_pct = ((hit.price - float(p.entry_price)) / float(p.entry_price)) * 100
            out.append(hit)

    return out


@app.get("/")
def root():
    """صفحة رئيسية بسيطة لحل مشكلة 404 من UptimeRobot"""
    return {
        "app": "Artoro API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "bot_status": "/bot/status",
            "bot_performance": "/bot/performance",
            "admin": "/admin",
            "docs": "/docs"
        },
        "message": "Welcome to Artoro API! Visit /docs for API documentation."
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/models/local")
def list_local_models():
    try:
        # Prefer the richer metadata format used by the admin UI.
        from api.routers.admin import list_local_models as _list_local_models
        return _list_local_models()
    except Exception as e:
        print(f"Warning: Failed to use admin router for local models: {e}")
        api_dir = os.path.dirname(os.path.abspath(__file__))

        models_dir = os.path.join(api_dir, "models")
        if not os.path.exists(models_dir):
            return {"models": []}

        files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
        files.sort()
        return {"models": files}


@app.get("/symbols/inventory")
def symbols_inventory():
    """Returns mapping of countries/exchanges to symbol/price counts."""
    from api.stock_ai import get_supabase_inventory
    return {"inventory": get_supabase_inventory()}

@app.get("/symbols/countries")
def symbols_countries(source: str = Query(default="supabase")):
    try:
        if source == "local":
            return {"countries": list_countries()}
        
        from api.stock_ai import get_supabase_countries
        sb_countries = get_supabase_countries()
        if sb_countries:
            return {"countries": sb_countries}
        
        return {"countries": list_countries()} # Fallback
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/symbols/by-date")
def symbols_by_date(
    request: Request,
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    exchange: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=5000),
    search_term: Optional[str] = Query(default=None),
):
    from api.stock_ai import _init_supabase, supabase

    def _chunks(items: list, size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    _init_supabase()
    if not supabase:
        return {"results": []}

    try:
        # Use RPC to efficiently get unique symbols with price data in the date range
        # RPC function get_exchange_symbols_prices:
        # SELECT DISTINCT ON (symbol) symbol, MAX(date) as last_date, MIN(date) as first_date, COUNT(*) as count FROM stock_prices
        # WHERE exchange = p_exchange AND date >= p_start AND date <= p_end
        # GROUP BY symbol ORDER BY symbol, date DESC;

        rpc_args = {
            "p_exchange": exchange,
            "p_start": start,
            "p_end": end,
            "p_limit": limit * 5 # Fetch more initially for filtering
        }

        # If exchange is None, this RPC won't work well without modifications.
        # For now, we assume exchange is always provided as per frontend logic.
        if not exchange:
            return {"results": []}

        rpc_res = supabase.rpc("get_exchange_symbols_prices", rpc_args).execute()
        if not rpc_res.data:
            return {"results": []}

        # rpc_res.data will contain unique symbols along with their date range counts.
        # We need to transform this into the format expected by the frontend.
        symbols_from_db = []
        for row in rpc_res.data:
            symbols_from_db.append({
                "symbol": row["symbol"],
                "exchange": exchange, # Exchange is implicit from the RPC call
                "name": "", # Will be enriched later
                "last_date": row["last_date"],
                "first_date": row["first_date"],
                "row_count": row["count"]
            })

        # Enrich with names from stock_fundamentals in chunks
        symbols_to_process = []
        if symbols_from_db:
            names_map: dict[str, str] = {}
            symbol_list = [s["symbol"] for s in symbols_from_db]
            
            for chunk in _chunks(symbol_list, 500):
                res = (
                    supabase.table("stock_fundamentals")
                    .select("symbol,data")
                    .in_("symbol", chunk)
                    .eq("exchange", exchange) # Filter fundamentals by exchange too
                    .execute()
                )
                if res.data:
                    for r in res.data:
                        d = r.get("data") or {}
                        names_map[r.get("symbol")] = d.get("name", d.get("Name", ""))

            for s in symbols_from_db:
                s["name"] = names_map.get(s["symbol"], "")
                symbols_to_process.append(s)

        # Apply search_term filter if provided
        if search_term:
            search_term_lower = search_term.lower()
            symbols_to_process = [
                s for s in symbols_to_process 
                if search_term_lower in s["name"].lower() or search_term_lower in s["symbol"].lower()
            ]
        
        # Apply limit after all filtering and sorting
        symbols_to_process = symbols_to_process[:limit]
        symbols_to_process.sort(key=lambda x: x["symbol"])
        return {"results": symbols_to_process}
    except Exception as e:
        print(f"Error in symbols_by_date: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/symbols/synced")
def symbols_synced(
    country: Optional[str] = Query(default=None),
    source: str = Query(default="supabase")
):
    """API for frontend to fetch all synced symbols once and cache."""
    try:
        if source == "local" and country:
            from api.symbols_local import load_symbols_for_country
            from api.stock_ai import is_ticker_synced, _init_supabase
            _init_supabase()
            raw = load_symbols_for_country(country)

            # Batch check for Supabase presence to avoid O(N) queries
            from api.stock_ai import batch_check_local_cache
            symbol_ex_list = []
            for r in raw:
                s = r.get("Symbol") or r.get("symbol") or r.get("Code") or r.get("code")
                ex = r.get("Exchange") or r.get("exchange")
                if s and ex:
                    symbol_ex_list.append((s, ex))
            
            sync_status = batch_check_local_cache(symbol_ex_list)

            # Map to consistent format
            results = []
            for r in raw:
                s = r.get("Symbol") or r.get("symbol") or r.get("Code") or r.get("code")
                ex = r.get("Exchange") or r.get("exchange")
                n = r.get("Name") or r.get("name") or ""
                if s and ex:
                    results.append({
                        "symbol": s,
                        "exchange": ex,
                        "name": n,
                        "country": country,
                        "hasLocal": sync_status.get((s, ex), False)
                    })
            return {"results": results}

        from api.stock_ai import get_supabase_symbols
        results = get_supabase_symbols(country=country)

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/symbols/search")
def symbols_search(
    q: str = Query(default="", max_length=64),
    country: str | None = Query(default=None, max_length=64),
    exchange: str | None = Query(default=None, max_length=24),
    limit: int = Query(default=25, ge=1, le=100000),
    source: str = Query(default="supabase")
):
    try:
        if source == "local":
            results = search_symbols(q=q, country=country, exchange=exchange, limit=limit)
            return {"results": results}
        
        # Supabase search
        from api.stock_ai import get_supabase_symbols
        all_sb = get_supabase_symbols(country=country)
        
        q_low = q.lower().strip()
        results = []
        for s in all_sb:
            s_name = str(s.get('name') or '')
            s_symbol = str(s.get('symbol') or '')
            if not q_low or q_low in s_symbol.lower() or q_low in s_name.lower():
                # Apply exchange filter if provided
                s_exchange = str(s.get('exchange') or '')
                if exchange and s_exchange.lower() != exchange.lower():
                    continue
                results.append(s)
                if len(results) >= limit:
                    break
        
        # If supabase has no results, maybe fallback to local or return empty
        # but user specifically asked to use supabase for the app.
        return {"results": results}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from threading import Lock
import time

_PREDICT_CACHE: Dict[str, Dict[str, Any]] = {}
_PREDICT_CACHE_LOCK = Lock()
_PREDICT_CACHE_TTL = 300 # 5 minutes

@app.post("/predict")
def predict(req: PredictRequest):
    # 1. Generate a cache key based on most important request fields
    cache_key = f"{req.ticker.strip().upper()}_{req.exchange or 'AUTO'}_{req.model_name or 'DEFAULT'}_{req.rf_preset}_{req.from_date}_{req.to_date}_{req.target_pct}_{req.stop_loss_pct}_{req.look_forward_days}_{req.buy_threshold}_{req.use_volatility_label}"
    
    # 2. Check cache
    with _PREDICT_CACHE_LOCK:
        cached = _PREDICT_CACHE.get(cache_key)
        if cached and (time.time() - cached["ts"] < _PREDICT_CACHE_TTL):
            return cached["data"]

    api_key = os.getenv("EODHD_API_KEY")
    if (not req.force_local) and (not api_key):
        raise HTTPException(status_code=500, detail="EODHD_API_KEY is not configured")

    try:
        payload = run_pipeline(
            api_key=api_key or "",
            ticker=req.ticker.strip().upper(),
            from_date=req.from_date,
            to_date=req.to_date,
            include_fundamentals=req.include_fundamentals,
            exchange=req.exchange,
            force_local=req.force_local,
            rf_preset=req.rf_preset,
            rf_params=req.rf_params,
            model_name=req.model_name,
            target_pct=req.target_pct,
            stop_loss_pct=req.stop_loss_pct,
            look_forward_days=req.look_forward_days,
            buy_threshold=req.buy_threshold,
            use_volatility_label=req.use_volatility_label,
        )
        
        # 3. Store in cache
        with _PREDICT_CACHE_LOCK:
            _PREDICT_CACHE[cache_key] = {
                "ts": time.time(),
                "data": payload
            }
            
        return payload
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Internal error in /predict: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/price")
def get_price(
    ticker: str = Query(default="", min_length=1, max_length=24, pattern=r"^[A-Za-z0-9.\-]{1,24}$"),
):
    t = ticker.strip().upper()
    cfg = _load_admin_config()
    source = (cfg.get("source") or "eodhd").lower()

    api_key = os.getenv("EODHD_API_KEY")
    as_of = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

    try:
        if source == "eodhd" and api_key:
            price = _fetch_price_eodhd(t, api_key)
            return {"ticker": t, "price": price, "source": "eodhd", "asOf": as_of}

        price = _fetch_price_yahoo(t)
        return {"ticker": t, "price": price, "source": "yahoo", "asOf": as_of}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Internal error in /price: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/news")
def get_news(symbol: str = Query(default="all_symbols")):
    try:
        # Use a general ticker for main news if none specified
        ticker_str = "SPY" if symbol == "all_symbols" else symbol
        yf_ticker = _normalize_yahoo_ticker(ticker_str)
        t = yf.Ticker(yf_ticker)
        
        raw_news = getattr(t, "news", [])
        articles = []
        
        for n in raw_news:
            # Map Yahoo fields to a standard format
            articles.append({
                "title": n.get("title"),
                "url": n.get("link"),
                "source": {"name": n.get("publisher", "Yahoo Finance")},
                "publishedAt": dt.datetime.fromtimestamp(n.get("providerPublishTime")).isoformat() if n.get("providerPublishTime") else None,
                "description": n.get("type", "Market News"), # Yahoo news rarely has full description in this API
                "image": n.get("thumbnail", {}).get("resolutions", [{}])[0].get("url") if n.get("thumbnail") else None
            })
            
        return {"articles": articles}
    except Exception as e:
        print(f"News fetch error: {e}")
        return {"articles": [], "error": str(e)}


# ------------------------------------------------------------------
# Backtest Endpoint
# ------------------------------------------------------------------
from pydantic import BaseModel as PBM

class BacktestRequest(PBM):
    exchange: str
    model: str
    start_date: str = "2024-01-01"
    end_date: str | None = None
    council_model: str | None = None
    validator_model: str | None = None
    meta_threshold: float | None = None
    council_threshold: float | None = None
    target_pct: float | None = None
    stop_loss_pct: float | None = None
    capital: float = 100000


def _safe_basename(name: str) -> str:
    # Prevent path traversal and accidental directory usage in user-provided model names.
    name = (name or "").strip()
    name = name.replace("\\", "/")
    return name.split("/")[-1]


def _available_local_models(models_dir: str) -> list[str]:
    try:
        names = []
        for fn in os.listdir(models_dir):
            if fn.lower().endswith(".pkl"):
                names.append(fn)
        return sorted(names)
    except Exception:
        return []

def _load_model_card(models_dir: str, model_name: str) -> dict | None:
    try:
        p = os.path.join(models_dir, f"{model_name}.model_card.json")
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _compute_benchmark_metrics(project_root: str, model_name: str, start_date: str, end_date: str | None, exchange: str | None = None) -> tuple[float | None, float | None, str | None]:
    """
    Returns (benchmark_return_pct, benchmark_win_rate, benchmark_name).
    Uses local index JSON referenced by the model card.
    """
    try:
        models_dir = os.path.join(project_root, "api", "models")
        card = _load_model_card(models_dir, model_name)
        index_rel = None
        if isinstance(card, dict):
            index_rel = (card.get("data_inputs") or {}).get("exchange_index_json_path")

        # Basic fallback for EGX models if the card is missing.
        ex = (exchange or (card or {}).get("exchange") or "").upper()
        if not index_rel and ex == "EGX":
            index_rel = os.path.join("symbols_data", "EGX30-INDEX.json")

        if not index_rel:
            # print(f"[BT-LIVE] DEBUG: No exchange index json path for model={model_name}, exchange={exchange}", flush=True)
            return None, None, None

        index_path = os.path.join(project_root, index_rel)
        # print(f"[BT-LIVE] DEBUG: Loading exchange index json for model={model_name}, exchange={exchange}, path={index_path}", flush=True)
        if not os.path.exists(index_path):
            return None, None, None

        with open(index_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list) or not rows:
            return None, None, None

        df = pd.DataFrame(rows)
        if "date" not in df.columns or "close" not in df.columns:
            return None, None, None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")

        sd = pd.to_datetime(start_date, errors="coerce")
        ed = pd.to_datetime(end_date, errors="coerce") if end_date else None
        if pd.isna(sd):
            return None, None, None
            
        # Filter for the simulation period
        mask = (df["date"] >= sd)
        if ed is not None and not pd.isna(ed):
            mask = mask & (df["date"] <= ed)
        
        period_df = df[mask]
        
        # Calculate Return
        start_row = df.loc[df["date"] >= sd].head(1)
        if ed is not None and not pd.isna(ed):
            end_row = df.loc[df["date"] <= ed].tail(1)
        else:
            end_row = df.tail(1)
            
        benchmark_return_pct = None
        if not start_row.empty and not end_row.empty:
            start_close = float(start_row["close"].iloc[0])
            end_close = float(end_row["close"].iloc[0])
            if (np.isfinite(start_close) and np.isfinite(end_close)) and start_close != 0:
                benchmark_return_pct = ((end_close / start_close) - 1.0) * 100.0

        # Calculate Win Rate (Positive daily returns)
        benchmark_win_rate = None
        if not period_df.empty and len(period_df) > 1:
            period_df = period_df.copy()
            # Calculate % change from previous close
            period_df["pct_change"] = period_df["close"].pct_change()
            # Drop the first row which is NaN
            period_df = period_df.dropna(subset=["pct_change"])
            
            if not period_df.empty:
                positive_days = len(period_df[period_df["pct_change"] > 0])
                total_days = len(period_df)
                benchmark_win_rate = (positive_days / total_days) * 100.0

        benchmark_name = os.path.splitext(os.path.basename(index_path))[0]
        # print(f"[BT-LIVE] DEBUG: Benchmark stats model={model_name}, exchange={exchange}, return_pct={benchmark_return_pct}, win_rate={benchmark_win_rate}, name={benchmark_name}", flush=True)
        return benchmark_return_pct, benchmark_win_rate, str(benchmark_name)
    except Exception:
        return None, None, None

@app.post("/backtest")
async def backtest_endpoint(req: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run backtest simulation as a background task to avoid timeouts.
    """
    # Validate the model name early to avoid expensive work and noisy background failures.
    api_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(api_dir, "models")
    requested_model = _safe_basename(req.model)
    model_path = os.path.join(models_dir, requested_model)
    if not os.path.exists(model_path):
        # Provide a helpful hint with closest matches.
        import difflib

        available = _available_local_models(models_dir)
        suggestions = difflib.get_close_matches(requested_model, available, n=5, cutoff=0.1)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "model_not_found",
                "model": requested_model,
                "message": f"Model not found in {models_dir}",
                "suggestions": suggestions,
            },
        )

    # Optional validator model (Council Validator)
    requested_validator = None
    if req.validator_model:
        requested_validator = _safe_basename(req.validator_model)
        validator_path = os.path.join(models_dir, requested_validator)
        if not os.path.exists(validator_path):
            import difflib

            available = _available_local_models(models_dir)
            suggestions = difflib.get_close_matches(requested_validator, available, n=5, cutoff=0.1)
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "validator_model_not_found",
                    "model": requested_validator,
                    "message": f"Validator model not found in {models_dir}",
                    "suggestions": suggestions,
                },
            )

    # 1. Create a placeholder record in Supabase to track status
    from api.stock_ai import supabase
    try:
        # Use today as default end_date if none provided
        end_date = req.end_date or dt.datetime.utcnow().date().isoformat()
        
        res = supabase.table("backtests").insert({
            "model_name": req.model,
            "exchange": req.exchange,
            "council_model": req.council_model,
            "start_date": req.start_date,
            "end_date": end_date,
            "status": "pending",
            "total_trades": 0,
            "win_rate": 0,
            "net_profit": 0,
            "avg_return_per_trade": 0,
            "meta_threshold": req.meta_threshold,
            "council_threshold": req.council_threshold,
            "target_pct": req.target_pct,
            "stop_loss_pct": req.stop_loss_pct,
            "capital": req.capital
        }).execute()
        
        backtest_id = res.data[0]["id"] if res.data else None
    except Exception as e:
        print(f"Error creating backtest record: {e}")
        backtest_id = None

    # Use the sanitized model name end-to-end (subprocess + model card lookup).
    req_sanitized = BacktestRequest(
        exchange=req.exchange,
        model=requested_model,
        start_date=req.start_date,
        end_date=req.end_date,
        council_model=req.council_model,
        validator_model=requested_validator,
        meta_threshold=req.meta_threshold,
        council_threshold=req.council_threshold,
        target_pct=req.target_pct,
        stop_loss_pct=req.stop_loss_pct,
        capital=req.capital,
    )

    background_tasks.add_task(run_backtest_task, req_sanitized, backtest_id)
    return {
        "status": "queued", 
        "id": backtest_id,
        "message": f"Backtest for {req.model} on {req.exchange} has been started. Trace ID: {backtest_id}"
    }

def run_backtest_task(req: BacktestRequest, backtest_id: str = None):
    """Internal task runner for backtests with real-time status updates."""
    import subprocess
    import csv
    import json
    import os
    import sys
    import datetime as dt
    from api.stock_ai import supabase
    
    model_name = req.model
    exchange = req.exchange
    start_date = req.start_date
    end_date = req.end_date or dt.datetime.utcnow().date().isoformat()
    
    # Build command
    api_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(api_dir)
    script_path = os.path.join(api_dir, "backtest_radar.py")
    
    cmd = [
        sys.executable, script_path,
        "--exchange", exchange,
        "--model", model_name,
        "--start", start_date,
        "--end", end_date
    ]
    
    if req.council_model:
        cmd.extend(["--council", req.council_model])

    if req.validator_model:
        cmd.extend(["--validator", req.validator_model])

    if req.meta_threshold is not None:
        cmd.extend(["--meta-threshold", str(req.meta_threshold)])
    
    if req.council_threshold is not None:
        cmd.extend(["--validator-threshold", str(req.council_threshold)])

    if req.target_pct is not None:
        cmd.extend(["--target-pct", str(req.target_pct)])

    if req.stop_loss_pct is not None:
        cmd.extend(["--stop-loss-pct", str(req.stop_loss_pct)])

    if req.capital is not None:
        cmd.extend(["--capital", str(req.capital)])
    
    # Always use quiet mode in background tasks to keep terminal clean
    cmd.append("--quiet")
    
    if not os.path.exists(script_path):
        print(f"Error: Backtest script not found at {script_path}")
        return
    
    try:
        print(f"Background Backtest Started: {model_name} on {exchange} (ID: {backtest_id})")
        
        # Update status to processing
        if backtest_id:
            try: supabase.table("backtests").update({"status": "processing", "status_msg": "Starting subprocess..."}).eq("id", backtest_id).execute()
            except: pass

        csv_path = os.path.join(api_dir, f"backtest_results_{exchange}_{backtest_id or 'latest'}.csv")
        cmd.extend(["--out", csv_path])

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=api_dir,
            encoding='utf-8',
            errors='replace',
            bufsize=1, # Line buffered
            universal_newlines=True
        )
        
        stdout_lines = []
        stderr_lines = []

        # Read stdout in real-time
        is_json_block = False
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                clean_line = line.strip()
                stdout_lines.append(line)
                
                # Filter out the huge JSON trades log from the terminal output
                if "--- JSON TRADES LOG START ---" in clean_line:
                    is_json_block = True
                    print(f"[BT-LIVE] {clean_line} (Suppressed in terminal)")
                elif "--- JSON TRADES LOG END ---" in clean_line:
                    is_json_block = False
                    print(f"[BT-LIVE] {clean_line}")
                elif not is_json_block:
                    print(f"[BT-LIVE] {clean_line}")
                
                # Update status if interesting progress found
                if backtest_id and any(x in clean_line for x in ["Fetching", "Progress:", "Loading", "Processing"]):
                    try:
                        # Extract "Progress: 60/246" or just use the line
                        msg = clean_line
                        if "Progress:" in clean_line:
                            msg = clean_line.split("...") [0].strip()
                        
                        supabase.table("backtests").update({"status_msg": msg}).eq("id", backtest_id).execute()
                    except:
                        pass

        # Capture remaining stderr
        for line in process.stderr:
            stderr_lines.append(line)
            print(f"[BT-ERR] {line.strip()}")

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        
        # Log to server console for finality
        print(f"--- Backtest Finished [{model_name}] ---")
        
        # Simple extraction logic (keep as-is or improve)
        total_trades = 0
        win_rate = 0.0
        net_profit = 0
        avg_return = 0.0
        trades = []
        
        lines = stdout.split("\n")
        pre_council_trades = 0
        post_council_trades = 0
        pre_council_win_rate = 0.0
        post_council_win_rate = 0.0
        pre_council_profit_pct = None
        post_council_profit_pct = None
        
        for line in lines:
            if "Total Trades Detected" in line:
                try: total_trades = int(line.split(":")[1].strip())
                except: pass
            elif "Win Rate:" in line and "Pre-Council" not in line and "Post-Council" not in line:
                try: win_rate = float(line.split(":")[1].strip().replace("%", ""))
                except: pass
            elif "Simulated Profit" in line or "Net Profit" in line:
                try:
                    parts = line.split(":")
                    if len(parts) > 1:
                        # Extract number, handle commas and currency suffix
                        clean_val = parts[1].strip().split(" ")[0].replace(",", "")
                        net_profit = int(float(clean_val))
                except: pass
            elif "Avg Return" in line:
                try: avg_return = float(line.split(":")[1].strip().replace("%", ""))
                except: pass
            elif "Pre-Council Trades:" in line:
                try: pre_council_trades = int(line.split(":")[1].strip())
                except: pass
            elif "Post-Council Trades:" in line:
                try: post_council_trades = int(line.split(":")[1].strip())
                except: pass
            elif "Pre-Council Win Rate:" in line:
                try: pre_council_win_rate = float(line.split(":")[1].strip().replace("%", ""))
                except: pass
            elif "Post-Council Win Rate:" in line:
                try: post_council_win_rate = float(line.split(":")[1].strip().replace("%", ""))
                except: pass
            elif "Pre-Council Profit:" in line:
                try: pre_council_profit_pct = float(line.split(":")[1].strip().replace("%", ""))
                except: pass
            elif "Post-Council Profit:" in line:
                try: post_council_profit_pct = float(line.split(":")[1].strip().replace("%", ""))
                except: pass
            elif "Rejected Profitable:" in line:
                try: rejected_profitable = int(line.split(":")[1].strip())
                except: pass
        
        # Parse JSON Trades Log from stdout
        try:
            val_start = stdout.find("--- JSON TRADES LOG START ---")
            val_end = stdout.find("--- JSON TRADES LOG END ---")
            if val_start != -1 and val_end != -1:
                json_str = stdout[val_start + len("--- JSON TRADES LOG START ---"):val_end].strip()
                parsed_trades = json.loads(json_str)
                for row in parsed_trades:
                    trades.append({
                        "date": row.get("Date", ""),
                        "symbol": row.get("Symbol", ""),
                        "entry": float(row.get("Entry", 0) or 0),
                        "exit": float(row.get("Exit", 0) or 0),
                        "result": row.get("Result", ""),
                        "pnl_pct": float(row.get("PnL_Pct", 0) or 0),
                        "status": row.get("Status", "Accepted"),
                        "votes": row.get("Votes", {}), # JSON keeps it as dict if it was dict
                        "Entry_Date": row.get("Entry_Date", ""),
                        "Exit_Date": row.get("Exit_Date", ""),
                        "Entry_Day": row.get("Entry_Day", ""),
                        "Exit_Day": row.get("Exit_Day", ""),
                        "Profit_Cash": float(row.get("Profit_Cash", 0) or 0),
                        "Cumulative_Profit": float(row.get("Cumulative_Profit", 0) or 0),
                        "Position_Cash": float(row.get("Position_Cash", 0) or 0),
                        "Size_Multiplier": float(row.get("Size_Multiplier", 0) or 0),
                        "Score": (float(row.get("Score")) if row.get("Score") is not None else None),
                        "Radar_Score": (float(row.get("Radar_Score")) if row.get("Radar_Score") is not None else None),
                        "Validator_Score": (float(row.get("Validator_Score")) if row.get("Validator_Score") is not None else None),
                        "Sizing_Score": (float(row.get("Sizing_Score")) if row.get("Sizing_Score") is not None else None),
                        "Fund_Score": (float(row.get("Fund_Score")) if row.get("Fund_Score") is not None else None),
                    })
        except Exception as e:
            print(f"Error parsing trades JSON: {e}")
        
        # Save to Supabase
        from api.stock_ai import _init_supabase, supabase
        _init_supabase()
        if supabase:
            # Compute total return % on a fixed notional capital.
            initial_capital = 100000.0
            profit_pct = None
            try:
                profit_pct = (float(net_profit) / float(initial_capital)) * 100.0
            except Exception:
                profit_pct = None
            if post_council_profit_pct is not None:
                profit_pct = float(post_council_profit_pct)

            bench_pct, bench_win_rate, bench_name = _compute_benchmark_metrics(
                project_root=project_root,
                model_name=model_name,
                start_date=start_date,
                end_date=end_date,
                exchange=exchange,
            )
            
            # Note: We are discarding Alpha Pct calculation and replacing it with Index Win Rate
            # as requested by the user.

            # Backtest results are stored only in the backtests table (trades_log).
            # We do not write backtest trades into scan_results.

            # 5. Final Save to Supabase
            if backtest_id:
                try:
                    update_payload = {
                        "status": "completed",
                        "status_msg": "Simulation finished successfully.",
                        "total_trades": total_trades,
                        "win_rate": win_rate,
                        "net_profit": net_profit,
                        "avg_return_per_trade": avg_return,
                        "trades_log": trades,
                        "council_model": req.council_model
                    }

                    # Optional new analytics columns (tolerate missing DB migration).
                    if profit_pct is not None:
                        update_payload["profit_pct"] = profit_pct
                    if bench_pct is not None:
                        update_payload["benchmark_return_pct"] = bench_pct
                    if bench_win_rate is not None:
                         # Store index win rate (replacing alpha_pct or as new field?)
                         # Reusing alpha_pct field might be confusing, assuming user will handle schema. 
                         # But user said "replace", so in UI it will replace.
                         # In DB, let's try to save as "benchmark_win_rate" if column exists
                         update_payload["benchmark_win_rate"] = bench_win_rate
                         
                    if bench_name:
                        update_payload["benchmark_name"] = bench_name

                    # Council analytics
                    if pre_council_trades or post_council_trades:
                        update_payload["pre_council_trades"] = pre_council_trades
                        update_payload["post_council_trades"] = post_council_trades
                        update_payload["pre_council_win_rate"] = pre_council_win_rate
                        update_payload["post_council_win_rate"] = post_council_win_rate
                        if pre_council_profit_pct is not None:
                            update_payload["pre_council_profit_pct"] = pre_council_profit_pct
                        if post_council_profit_pct is not None:
                            update_payload["post_council_profit_pct"] = post_council_profit_pct
                        if 'rejected_profitable' in locals():
                             # Schema doesn't have this column yet
                             pass

                    try:
                        # Clear alpha_pct if it existed
                        update_payload["alpha_pct"] = None 
                        supabase.table("backtests").update(update_payload).eq("id", backtest_id).execute()
                    except Exception:
                        # Retry without optional columns
                        for k in ("profit_pct", "benchmark_return_pct", "benchmark_win_rate", "benchmark_name", "alpha_pct"):
                            update_payload.pop(k, None)
                        supabase.table("backtests").update(update_payload).eq("id", backtest_id).execute()

                    print(f"Background Backtest Updated & Saved: {model_name} (ID: {backtest_id})")
                except Exception as e:
                    print(f"Error updating backtest result: {e}")
            else:
                # Fallback to old behavior if no ID (shouldn't happen now)
                try:
                    # Use today as default end_date if none provided
                    final_end_date = req.end_date or dt.datetime.utcnow().date().isoformat()
                    
                    supabase.table("backtests").insert({
                        "model_name": model_name,
                        "exchange": exchange,
                        "council_model": req.council_model,
                        "start_date": req.start_date,
                        "end_date": final_end_date,
                        "total_trades": total_trades,
                        "win_rate": win_rate,
                        "net_profit": net_profit,
                        "avg_return_per_trade": avg_return,
                        "trades_log": trades,
                        "status": "completed",
                        "profit_pct": profit_pct,
                        "benchmark_return_pct": bench_pct,
                        "benchmark_name": bench_name,
                        "pre_council_trades": pre_council_trades if pre_council_trades > 0 else None,
                        "post_council_trades": post_council_trades if pre_council_trades > 0 else None,
                        "pre_council_win_rate": pre_council_win_rate if pre_council_trades > 0 else None,
                        "post_council_win_rate": post_council_win_rate if pre_council_trades > 0 else None,
                        "pre_council_profit_pct": pre_council_profit_pct if 'pre_council_profit_pct' in locals() and pre_council_profit_pct is not None else None,
                        "post_council_profit_pct": post_council_profit_pct if 'post_council_profit_pct' in locals() and post_council_profit_pct is not None else None
                    }).execute()
                    print(f"Background Backtest Saved (Fallback): {model_name}")
                except Exception as e:
                    print(f"Error saving backtest result: {e}")

    except Exception as e:
        print(f"Backtest Task Failed: {e}")
        if backtest_id:
            try: supabase.table("backtests").update({"status": "failed", "status_msg": str(e)}).eq("id", backtest_id).execute()
            except: pass


@app.get("/backtests")
async def get_backtests(model: Optional[str] = None):
    """Fetch all backtest historical records."""
    import time as _time
    import api.stock_ai as stock_ai

    # Soft cache to keep UI stable if Supabase intermittently fails (e.g., SSL EOF during polling)
    global _BACKTESTS_SOFT_CACHE  # type: ignore[name-defined]
    try:
        _BACKTESTS_SOFT_CACHE
    except Exception:
        _BACKTESTS_SOFT_CACHE = {"ts": 0.0, "data": []}

    stock_ai._init_supabase()
    if not stock_ai.supabase:
        # Return cache instead of hard-failing when the UI polls aggressively
        return _BACKTESTS_SOFT_CACHE.get("data", [])

    def _build_query():
        q = stock_ai.supabase.table("backtests").select("*").order("created_at", desc=True)
        if model:
            q = q.eq("model_name", model)
        return q

    last_err = None
    for attempt in range(1, 4):
        try:
            res = _build_query().execute()
            data = res.data or []
            _BACKTESTS_SOFT_CACHE = {"ts": _time.time(), "data": data}
            return data
        except Exception as e:
            last_err = e
            # Re-init and retry (helps when client/connection gets into a bad state)
            try:
                stock_ai.supabase = None
            except Exception:
                pass
            stock_ai._init_supabase()
            _time.sleep(0.35 * attempt)

    print(f"Unhandled exception for GET /backtests: {last_err}")
    return _BACKTESTS_SOFT_CACHE.get("data", [])


@app.get("/backtests/{id}/trades")
async def get_backtest_trades(id: str):
    """Fetch trades for a given backtest (stored in scan_results)."""
    from api.stock_ai import supabase
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not initialized")

    fields = "symbol,exchange,model_name,entry_price,exit_price,profit_loss_pct,status,features,created_at"
    
    # Helper to map raw log to scan_results format
    def _map_trades_log(log):
        # Handle stringified JSON
        if isinstance(log, str):
            try:
                log = json.loads(log)
            except Exception:
                return []
                
        mapped = []
        if not log or not isinstance(log, list):
            return []
            
        for t in log:
            if not isinstance(t, dict): continue
            pnl = float(t.get("pnl_pct") or 0)
            mapped.append({
                "symbol": t.get("symbol"),
                "entry_price": float(t.get("entry") or 0),
                "exit_price": float(t.get("exit") or 0),
                "profit_loss_pct": pnl * 100.0 if pnl < 1.0 and pnl > -1.0 else pnl, # Handle both 0.05 and 5.0
                "status": "win" if pnl > 0 else "loss",
                "features": {
                    "trade_date": t.get("date"),
                    "backtest_status": t.get("status") or t.get("Status") or "Accepted", 
                    "votes": "{}",
                    "entry_date": t.get("Entry_Date"),
                    "exit_date": t.get("Exit_Date"),
                    "entry_day": t.get("Entry_Day"),
                    "exit_day": t.get("Exit_Day"),
                    "profit_cash": t.get("Profit_Cash") or t.get("features", {}).get("profit_cash"),
                    "cumulative_profit": t.get("Cumulative_Profit") or t.get("features", {}).get("cumulative_profit"),
                    "ai_score": t.get("Score") or t.get("score") or t.get("features", {}).get("ai_score"),
                    "radar_score": t.get("Radar_Score") or t.get("radar_score") or t.get("features", {}).get("radar_score"),
                    "fund_score": t.get("Fund_Score") or t.get("fund_score") or t.get("features", {}).get("fund_score")
                },
                "created_at": t.get("date")
            })
        return mapped

    # 1. Try fetching from scan_results (preferred)
    try:
        res = supabase.table("scan_results").select(fields).eq("batch_id", id).eq("source", "backtest").execute()
        if res.data:
            return res.data
    except Exception:
        # Fallback if source column isn't available yet
        try:
            res = supabase.table("scan_results").select(fields).eq("batch_id", id).execute()
            if res.data:
                return res.data
        except Exception:
            pass
            
    # 2. Fallback to backtests table trades_log
    # This acts as the final source of truth for backtests that haven't been synced to scan_results
    try:
        bt_res = supabase.table("backtests").select("trades_log").eq("id", id).execute()
        if bt_res.data and bt_res.data[0].get("trades_log"):
            return _map_trades_log(bt_res.data[0]["trades_log"])
    except Exception:
        pass

    return []


@app.delete("/backtests/{id}")
async def delete_backtest(id: str):
    """Delete a backtest record."""
    from api.stock_ai import supabase
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not initialized")
    
    res = supabase.table("backtests").delete().eq("id", id).execute()
    return {"status": "success", "deleted": id}


class BacktestUpdate(BaseModel):
    is_public: bool

@app.patch("/backtests/{id}")
async def update_backtest(id: str, req: BacktestUpdate):
    """Update visibility of a backtest record."""
    from api.stock_ai import supabase
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not initialized")
    
    res = supabase.table("backtests").update({"is_public": req.is_public}).eq("id", id).execute()
    return res.data[0] if res.data else {"error": "not found"}


