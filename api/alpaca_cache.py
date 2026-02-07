import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _project_root() -> str:
    # api/ is one level below project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def alpaca_exchanges_dir() -> str:
    return os.path.join(_project_root(), "alpaca_exchanges")


def _safe_read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)


def _safe_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    s = (name or "").strip().upper()
    s = _FILENAME_SAFE.sub("_", s)
    return s or "UNKNOWN"


def market_dir(market: str) -> str:
    return os.path.join(alpaca_exchanges_dir(), market)


def exchanges_cache_path(market: str) -> str:
    return os.path.join(market_dir(market), "exchanges.json")


def all_assets_cache_path(market: str) -> str:
    return os.path.join(market_dir(market), "all_assets.json")


def exchange_assets_cache_path(market: str, exchange: str) -> str:
    return os.path.join(market_dir(market), f"{_safe_filename(exchange)}.json")


def load_cached_exchanges(market: str) -> Optional[List[Dict[str, Any]]]:
    path = exchanges_cache_path(market)
    if not os.path.exists(path):
        return None
    data = _safe_read_json(path)
    return data if isinstance(data, list) else None


def load_cached_assets(market: str, exchange: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    if exchange:
        path = exchange_assets_cache_path(market, exchange)
        if not os.path.exists(path):
            return None
        data = _safe_read_json(path)
        return data if isinstance(data, list) else None

    path = all_assets_cache_path(market)
    if not os.path.exists(path):
        return None
    data = _safe_read_json(path)
    return data if isinstance(data, list) else None


def write_market_cache(
    market: str,
    assets: List[Dict[str, Any]],
    *,
    timestamp: Optional[str] = None,
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Writes:
      - alpaca_exchanges/<market>/all_assets.json
      - alpaca_exchanges/<market>/exchanges.json
      - alpaca_exchanges/<market>/<EXCHANGE>.json
    Returns (timestamp, total_count, exchanges_list)
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    mdir = market_dir(market)
    os.makedirs(mdir, exist_ok=True)

    by_exchange: Dict[str, List[Dict[str, Any]]] = {}
    for a in assets or []:
        ex = str(a.get("exchange") or a.get("Exchange") or "UNKNOWN").strip() or "UNKNOWN"
        by_exchange.setdefault(ex, []).append(a)

    exchanges = [{"name": ex, "asset_count": len(lst)} for ex, lst in sorted(by_exchange.items(), key=lambda kv: kv[0])]

    _safe_write_json(all_assets_cache_path(market), assets or [])
    _safe_write_json(exchanges_cache_path(market), exchanges)
    for ex, lst in by_exchange.items():
        _safe_write_json(exchange_assets_cache_path(market, ex), lst)

    # Also keep a tiny meta file for humans/debug
    _safe_write_json(
        os.path.join(mdir, "meta.json"),
        {"market": market, "updated_at": ts, "total_assets": len(assets or [])},
    )

    return ts, len(assets or []), exchanges

