import os
import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

def _project_root() -> str:
    # api/ symbols_local.py is one level below project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _default_symbols_dir() -> str:
    return os.path.join(_project_root(), "symbols_data")

def _safe_read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@lru_cache(maxsize=1)
def load_country_summary() -> Dict[str, Any]:
    base = _default_symbols_dir()
    path = os.path.join(base, "country_summary_20250304_171206.json")
    if not os.path.exists(path):
        return {}
    return _safe_read_json(path)

def list_countries() -> List[str]:
    summary = load_country_summary()
    return sorted(summary.keys())

def _country_file_name(country: str) -> str:
    return f"{country}_all_symbols_20250304_171206.json"

@lru_cache(maxsize=64)
def load_symbols_for_country(country: str) -> List[Dict[str, Any]]:
    base = _default_symbols_dir()
    path = os.path.join(base, _country_file_name(country))
    if not os.path.exists(path):
        return []
    data = _safe_read_json(path)
    return data if isinstance(data, list) else []

@lru_cache(maxsize=1)
def load_all_symbols() -> List[Dict[str, Any]]:
    base = _default_symbols_dir()
    path = os.path.join(base, "all_symbols_by_country_20250304_171206.json")
    if not os.path.exists(path):
        return []
    data = _safe_read_json(path)
    return data if isinstance(data, list) else []

def search_symbols(
    q: str,
    country: Optional[str] = None,
    exchange: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    query = (q or "").strip().lower()
    limit = max(1, min(int(limit), 100000))

    if country:
        haystack = load_symbols_for_country(country)
    else:
        haystack = load_all_symbols()

    from api.stock_ai import is_ticker_synced
    
    out = []
    ex_low = exchange.lower() if exchange else None

    for row in haystack:
        sym = str(row.get("Symbol", row.get("symbol", "")))
        name = str(row.get("Name", row.get("name", "")))
        ex = str(row.get("Exchange", row.get("exchange", "")))
        
        if ex_low and ex.lower() != ex_low: continue
        
        if not query or query in sym.lower() or query in name.lower():
            out.append({
                "symbol": sym,
                "exchange": ex,
                "name": name,
                "country": row.get("Country", row.get("country", "")),
                "hasLocal": is_ticker_synced(sym, ex)
            })
            if len(out) >= limit: break

    out.sort(key=lambda x: x.get("hasLocal", False), reverse=True)
    return out
