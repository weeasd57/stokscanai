import os
import sys
import json
import datetime as dt
import urllib.request
import pandas as pd
import warnings

# Suppress specific FutureWarnings from libraries like 'ta'
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the directory containing this file to sys.path to allow local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
# Explicitly load .env from the root directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))
load_dotenv(os.path.join(base_dir, "web", ".env.local"), override=True)

print(f"DEBUG: EODHD_API_KEY loaded: {'Yes' if os.getenv('EODHD_API_KEY') else 'No'}")

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, Optional, List, Literal

from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

import yfinance as yf

from stock_ai import run_pipeline
from symbols_local import list_countries, search_symbols
from routers import scan_ai, scan_ai_fast, scan_tech, admin

load_dotenv()

# Debug: Print loaded environment variables
print(f"DEBUG: NEXT_PUBLIC_SUPABASE_URL loaded: {'Yes' if os.getenv('NEXT_PUBLIC_SUPABASE_URL') else 'No'}")
print(f"DEBUG: SUPABASE_SERVICE_ROLE_KEY loaded: {'Yes' if os.getenv('SUPABASE_SERVICE_ROLE_KEY') else 'No'}")

app = FastAPI(title="AI Stocks API", version="1.0.0")

# Initialize Supabase at startup
from stock_ai import _init_supabase
_init_supabase()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    print(f"Unhandled exception for {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

app.include_router(scan_ai.router)
app.include_router(scan_ai_fast.router)
app.include_router(scan_tech.router)
app.include_router(admin.router)

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
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=24, pattern=r"^[A-Za-z0-9.\-]{1,24}$")
    exchange: Optional[str] = Field(default=None)
    from_date: str = Field(default="2020-01-01")
    include_fundamentals: bool = Field(default=True)
    rf_preset: Optional[str] = Field(default=None)
    rf_params: Optional[Dict[str, Any]] = Field(default=None)
    model_name: Optional[str] = Field(default=None)
    force_local: bool = Field(default=False)
    target_pct: float = Field(default=0.15)
    stop_loss_pct: float = Field(default=0.05)
    look_forward_days: int = Field(default=20)
    buy_threshold: float = Field(default=0.40)


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
        from stock_ai import _infer_symbol_exchange, get_stock_data_eodhd, _finite_float
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


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/models/local")
def list_local_models():
    try:
        # Prefer the richer metadata format used by the admin UI.
        from routers.admin import list_local_models as _list_local_models
        return _list_local_models()
    except Exception:
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
    from stock_ai import get_supabase_inventory
    return {"inventory": get_supabase_inventory()}

@app.get("/symbols/countries")
def symbols_countries(source: str = Query(default="supabase")):
    try:
        if source == "local":
            return {"countries": list_countries()}
        
        from stock_ai import get_supabase_countries
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
    from stock_ai import _init_supabase, supabase

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
            from symbols_local import load_symbols_for_country
            from stock_ai import is_ticker_synced, _init_supabase
            _init_supabase()
            raw = load_symbols_for_country(country)
            # Map to consistent format, handling potential case differences in JSON keys
            results = []
            for r in raw:
                # Try capitalized first (standard for these files), then lowercase, also check "Code"
                s = r.get("Symbol") or r.get("symbol") or r.get("Code") or r.get("code")
                ex = r.get("Exchange") or r.get("exchange")
                n = r.get("Name") or r.get("name") or ""
                if s and ex:
                    results.append({
                        "symbol": s,
                        "exchange": ex,
                        "name": n,
                        "country": country,
                        "hasLocal": is_ticker_synced(s, ex)
                    })
            return {"results": results}

        from stock_ai import get_supabase_symbols
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
        from stock_ai import get_supabase_symbols
        all_sb = get_supabase_symbols(country=country)
        
        q_low = q.lower().strip()
        results = []
        for s in all_sb:
            if not q_low or q_low in s['symbol'].lower() or q_low in s['name'].lower():
                # Apply exchange filter if provided
                if exchange and s['exchange'].lower() != exchange.lower():
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
    cache_key = f"{req.ticker.strip().upper()}_{req.exchange or 'AUTO'}_{req.model_name or 'DEFAULT'}_{req.rf_preset}_{req.from_date}_{req.target_pct}_{req.stop_loss_pct}_{req.look_forward_days}_{req.buy_threshold}"
    
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
