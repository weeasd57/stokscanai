import os
import json
from functools import lru_cache
from typing import Any, Dict, List, Optional
from stock_ai import check_local_cache


def _project_root() -> str:
    # api/ is one level below project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_symbols_dir() -> str:
    return os.path.join(_project_root(), "symbols_data")


def _safe_read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_country_summary(symbols_dir: Optional[str] = None) -> Dict[str, Any]:
    base = symbols_dir or _default_symbols_dir()
    path = os.path.join(base, "country_summary_20250304_171206.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"country summary not found: {path}")
    return _safe_read_json(path)


def list_countries(symbols_dir: Optional[str] = None) -> List[str]:
    summary = load_country_summary(symbols_dir)
    return sorted(summary.keys())


def _country_file_name(country: str) -> str:
    # Matches your naming convention: "Egypt_all_symbols_20250304_171206.json"
    return f"{country}_all_symbols_20250304_171206.json"


@lru_cache(maxsize=64)
def load_symbols_for_country(country: str, symbols_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    base = symbols_dir or _default_symbols_dir()
    path = os.path.join(base, _country_file_name(country))
    if not os.path.exists(path):
        raise FileNotFoundError(f"country symbols not found: {path}")
    data = _safe_read_json(path)
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected symbols JSON format")


@lru_cache(maxsize=1)
def load_all_symbols(symbols_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    base = symbols_dir or _default_symbols_dir()
    path = os.path.join(base, "all_symbols_by_country_20250304_171206.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"all symbols not found: {path}")
    data = _safe_read_json(path)
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected symbols JSON format")


def search_symbols(
    q: str,
    country: Optional[str] = None,
    exchange: Optional[str] = None,
    limit: int = 25,
    symbols_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = (q or "").strip()
    # if not query:
    #    return []  <-- ALLOW EMPTY QUERY to return all
    pass

    # Limit handling
    limit = max(1, min(int(limit), 10000))

    if country:
        haystack = load_symbols_for_country(country, symbols_dir)
    else:
        haystack = load_all_symbols(symbols_dir)

    q_low = query.lower()
    ex_low = exchange.lower() if exchange else None

    # Optimization: Pre-fetch cached tickers to avoid I/O in loop
    # We default cache_dir to standard location if not provided
    from stock_ai import get_cached_tickers, DEFAULT_CACHE_DIR, _safe_cache_key
    
    # Resolving cache dir (best effort)
    import time
    t0 = time.time()
    
    c_dir = os.getenv("CACHE_DIR", DEFAULT_CACHE_DIR)
    print(f"DEBUG: cache_dir={c_dir}")
    
    cached_set = get_cached_tickers(c_dir)
    t1 = time.time()
    print(f"DEBUG: get_cached_tickers took {t1-t0:.4f}s. Items: {len(cached_set)}")

    def _is_cached_fast(s: str, e: str) -> bool:
        # 1. Exact match
        key = _safe_cache_key(s)
        if key in cached_set: return True
        
        # 2. Exchange suffix match
        if e:
            mapping = {"EGX": "EGX", "US": "US", "NYSE": "US", "NASDAQ": "US"}
            suffix = mapping.get(e.upper(), e.upper())
            base = s.split('.')[0]
            cand = _safe_cache_key(f"{base}.{suffix}")
            if cand in cached_set: return True
            
        # 3. EGX legacy
        if "CC" in s or (e and e.upper() == "EGX"):
            base = s.split('.')[0]
            if _safe_cache_key(f"{base}.EGX") in cached_set: return True
            
        return False

    out: List[Dict[str, Any]] = []
    
    t_loop_start = time.time()
    for row in haystack:
        sym = str(row.get("Symbol", ""))
        name = str(row.get("Name", ""))
        ex = str(row.get("Exchange", ""))
        ctry = str(row.get("Country", ""))

        if ex_low and ex.lower() != ex_low:
            continue

        # Allow empty query to return all (limited by limit)
        if not query or sym.lower().startswith(q_low) or q_low in name.lower():
            # Check fast in-memory set
            has_local = _is_cached_fast(sym, ex)
            out.append({
                "symbol": sym, 
                "exchange": ex, 
                "name": name, 
                "country": ctry,
                "hasLocal": has_local
            })
            if len(out) >= limit:
                break
    t_loop_end = time.time()
    print(f"DEBUG: Search loop took {t_loop_end-t_loop_start:.4f}s. Results: {len(out)}")

    # Sort results to put local ones first
    out.sort(key=lambda x: x.get("hasLocal", False), reverse=True)

    return out

    # Sort results to put local ones first
    out.sort(key=lambda x: x.get("hasLocal", False), reverse=True)

    return out
