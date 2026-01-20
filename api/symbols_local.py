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
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Retry with BOM-tolerant decoding (utf-8-sig) for legacy JSON exports
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

def _find_latest_file(prefix: str, suffix: str = ".json") -> Optional[str]:
    base = _default_symbols_dir()
    if not os.path.exists(base):
        return None
    
    # Standard names first (if we decide to use them)
    standard = os.path.join(base, f"{prefix}{suffix}")
    if os.path.exists(standard):
        return standard
        
    # Otherwise look for timestamped ones: prefix_YYYYMMDD_HHMMSS.json
    candidates = [f for f in os.listdir(base) if f.startswith(prefix) and f.endswith(suffix)]
    if not candidates:
        return None
        
    # Sort by name (timestamp format ensures correct sorting for latest)
    candidates.sort(reverse=True)
    return os.path.join(base, candidates[0])

@lru_cache(maxsize=1)
def load_country_summary() -> Dict[str, Any]:
    path = _find_latest_file("country_summary")
    if not path:
        return {}
    return _safe_read_json(path)

def list_countries() -> List[str]:
    summary = load_country_summary()
    return sorted(summary.keys())

@lru_cache(maxsize=64)
def load_symbols_for_country(country: str) -> List[Dict[str, Any]]:
    path = _find_latest_file(f"{country}_all_symbols")
    if not path:
        return []
    data = _safe_read_json(path)
    return data if isinstance(data, list) else []

@lru_cache(maxsize=1)
def load_all_symbols() -> List[Dict[str, Any]]:
    path = _find_latest_file("all_symbols_by_country")
    if not path:
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
        sym = str(row.get("Symbol", row.get("symbol", row.get("Code", ""))))
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
