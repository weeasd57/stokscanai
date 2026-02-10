import os
import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

def _project_root() -> str:
    # 1. Start with the directory containing this file (api/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Check if symbols_data exists in the parent (typical local/Vercel structure)
    parent_dir = os.path.dirname(current_dir)
    if os.path.exists(os.path.join(parent_dir, "symbols_data")):
        return parent_dir
        
    # 3. Check /app (typical Hugging Face structure)
    if os.path.exists("/app/symbols_data"):
        return "/app"
        
    # 4. Fallback to parent directory as project root
    return parent_dir

def _default_symbols_dir() -> str:
    root = _project_root()
    path = os.path.join(root, "symbols_data")
    if not os.path.exists(path):
        print(f"DEBUG: symbols_data not found in {path}. Root was {root}")
    return path

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

    from api.stock_ai import batch_check_local_cache
    
    # 1. First pass: filter by query and exchange
    candidates = []
    ex_low = exchange.lower() if exchange else None
    for row in haystack:
        sym = str(row.get("Symbol", row.get("symbol", row.get("Code", ""))))
        name = str(row.get("Name", row.get("name", "")))
        ex = str(row.get("Exchange", row.get("exchange", "")))
        
        if ex_low and ex.lower() != ex_low: continue
        
        if not query or query in sym.lower() or query in name.lower():
            candidates.append((sym, ex, name, row.get("Country", row.get("country", ""))))
            # Since we sort by hasLocal later, we might need more than 'limit' 
            # to find the ones that ARE local. 
            # But normally we don't want to process 100k if limit is 50.
            # However, if limit is 100k, we process all.
            if len(candidates) >= limit * 2 and limit < 1000:
                break

    # 2. Batch check local status
    symbol_ex_list = [(c[0], c[1]) for c in candidates]
    sync_status = batch_check_local_cache(symbol_ex_list)

    # 3. Build final output
    out = []
    for sym, ex, name, country in candidates:
        out.append({
            "symbol": sym,
            "exchange": ex,
            "name": name,
            "country": country,
            "hasLocal": sync_status.get((sym, ex), False)
        })

    out.sort(key=lambda x: x.get("hasLocal", False), reverse=True)
    return out[:limit]
