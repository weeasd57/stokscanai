"""
TradingView Integration Module

Provides functions to fetch price history and fundamentals from TradingView.
Used by admin panel and stock_ai module for data updates.
"""

import os
import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


# Professional Exchange Configuration Mapping
# Inspired by stockroom model structures
EXCHANGE_CONFIG = {
    "ADX": {"market": "uae", "tv_id": "ADX", "country": "UAE"},
    "AS": {"market": "netherlands", "tv_id": "EURONEXT", "country": "Netherlands"},
    "AT": {"market": "greece", "tv_id": "ATHEX", "country": "Greece"},
    "AU": {"market": "australia", "tv_id": "ASX", "country": "Australia"},
    "BA": {"market": "argentina", "tv_id": "BCBA", "country": "Argentina"},
    "BC": {"market": "morocco", "tv_id": "CSE", "country": "Morocco"},
    "BE": {"market": "germany", "tv_id": "BER", "country": "Germany"},
    "BK": {"market": "thailand", "tv_id": "SET", "country": "Thailand"},
    "BR": {"market": "belgium", "tv_id": "EURONEXT", "country": "Belgium"},
    "BUD": {"market": "hungary", "tv_id": "BET", "country": "Hungary"},
    "CA": {"market": "egypt", "tv_id": "EGX", "country": "Egypt"},
    "CM": {"market": "sri_lanka", "tv_id": "CM", "country": "Sri Lanka"},
    "CO": {"market": "denmark", "tv_id": "OMXCOP", "country": "Denmark"},
    "DFM": {"market": "uae", "tv_id": "DFM", "country": "UAE"},
    "DSE": {"market": "tanzania", "tv_id": "DSE", "country": "Tanzania"},
    "DU": {"market": "germany", "tv_id": "DUS", "country": "Germany"},
    "EGX": {"market": "egypt", "tv_id": "EGX", "country": "Egypt"},
    "EUBOND": {"market": "belgium", "tv_id": "EUBOND", "country": "Belgium"},
    "F": {"market": "germany", "tv_id": "XETRA", "country": "Germany"},
    "GSE": {"market": "ghana", "tv_id": "GSE", "country": "Ghana"},
    "HA": {"market": "germany", "tv_id": "HA", "country": "Germany"},
    "HE": {"market": "finland", "tv_id": "OMXHEL", "country": "Finland"},
    "HM": {"market": "germany", "tv_id": "HM", "country": "Germany"},
    "IC": {"market": "iceland", "tv_id": "ICEX", "country": "Iceland"},
    "IL": {"market": "uk", "tv_id": "LSE", "country": "UK"},
    "IR": {"market": "ireland", "tv_id": "EURONEXT", "country": "Ireland"},
    "IS": {"market": "turkey", "tv_id": "BIST", "country": "Turkey"},
    "JK": {"market": "indonesia", "tv_id": "IDX", "country": "Indonesia"},
    "JSE": {"market": "south_africa", "tv_id": "JSE", "country": "South Africa"},
    "KAR": {"market": "pakistan", "tv_id": "KAR", "country": "Pakistan"},
    "KLSE": {"market": "malaysia", "tv_id": "MYX", "country": "Malaysia"},
    "KO": {"market": "korea", "tv_id": "KOSPI", "country": "Korea"},
    "KQ": {"market": "korea", "tv_id": "KOSDAQ", "country": "Korea"},
    "LIM": {"market": "peru", "tv_id": "LIM", "country": "Peru"},
    "LS": {"market": "portugal", "tv_id": "EURONEXT", "country": "Portugal"},
    "LSE": {"market": "uk", "tv_id": "LSE", "country": "UK"},
    "LU": {"market": "luxembourg", "tv_id": "LUXSE", "country": "Luxembourg"},
    "LUSE": {"market": "zambia", "tv_id": "LUSE", "country": "Zambia"},
    "MC": {"market": "spain", "tv_id": "BME", "country": "Spain"},
    "MSE": {"market": "malawi", "tv_id": "MSE", "country": "Malawi"},
    "MU": {"market": "germany", "tv_id": "MUN", "country": "Germany"},
    "MX": {"market": "mexico", "tv_id": "BMV", "country": "Mexico"},
    "NASDAQ": {"market": "america", "tv_id": "NASDAQ", "country": "USA"},
    "NEO": {"market": "canada", "tv_id": "NEO", "country": "Canada"},
    "NSE": {"market": "india", "tv_id": "NSE", "country": "India"},
    "NYSE": {"market": "america", "tv_id": "NYSE", "country": "USA"},
    "OL": {"market": "norway", "tv_id": "OSLO", "country": "Norway"},
    "PA": {"market": "france", "tv_id": "EURONEXT", "country": "France"},
    "PR": {"market": "czech", "tv_id": "PRAGUE", "country": "Czech Republic"},
    "PSE": {"market": "philippines", "tv_id": "PSE", "country": "Philippines"},
    "RO": {"market": "romania", "tv_id": "BVB", "country": "Romania"},
    "RSE": {"market": "rwanda", "tv_id": "RSE", "country": "Rwanda"},
    "SA": {"market": "brazil", "tv_id": "BMFBOVESPA", "country": "Brazil"},
    "SEM": {"market": "mauritius", "tv_id": "SEM", "country": "Mauritius"},
    "SHE": {"market": "china", "tv_id": "SZSE", "country": "China"},
    "SHG": {"market": "china", "tv_id": "SSE", "country": "China"},
    "SN": {"market": "chile", "tv_id": "SN", "country": "Chile"},
    "ST": {"market": "sweden", "tv_id": "OMXSTO", "country": "Sweden"},
    "STU": {"market": "germany", "tv_id": "STU", "country": "Germany"},
    "SW": {"market": "switzerland", "tv_id": "SIX", "country": "Switzerland"},
    "TO": {"market": "canada", "tv_id": "TSX", "country": "Canada"},
    "TW": {"market": "taiwan", "tv_id": "TWSE", "country": "Taiwan"},
    "TWO": {"market": "taiwan", "tv_id": "TPEX", "country": "Taiwan"},
    "US": {"market": "america", "tv_id": "NASDAQ", "country": "USA"},
    "USE": {"market": "uganda", "tv_id": "USE", "country": "Uganda"},
    "V": {"market": "canada", "tv_id": "TSXV", "country": "Canada"},
    "VFEX": {"market": "zimbabwe", "tv_id": "VFEX", "country": "Zimbabwe"},
    "VI": {"market": "austria", "tv_id": "VIE", "country": "Austria"},
    "VN": {"market": "vietnam", "tv_id": "HOSE", "country": "Vietnam"},
    "WAR": {"market": "poland", "tv_id": "GPW", "country": "Poland"},
    "XBOT": {"market": "botswana", "tv_id": "XBOT", "country": "Botswana"},
    "XETRA": {"market": "germany", "tv_id": "XETRA", "country": "Germany"},
    "XNAI": {"market": "kenya", "tv_id": "XNAI", "country": "Kenya"},
    "XNSA": {"market": "nigeria", "tv_id": "XNSA", "country": "Nigeria"},
    "XZIM": {"market": "zimbabwe", "tv_id": "XZIM", "country": "Zimbabwe"},
    "ZSE": {"market": "croatia", "tv_id": "ZSE", "country": "Croatia"},
}


def get_tradingview_market(symbol: str) -> str:
    """
    Get TradingView market name from symbol exchange suffix.
    
    Args:
        symbol: Stock symbol with exchange suffix (e.g., "AAPL.US", "AIR.PA")
    
    Returns:
        TradingView market name (e.g., "america", "france")
    """
    upper = (symbol or "").strip().upper()
    suffix = upper.split(".")[-1] if "." in upper else ""
    config = EXCHANGE_CONFIG.get(suffix)
    if config:
        return config["market"]
    return os.getenv("TRADINGVIEW_DEFAULT_MARKET", "america")


def get_tradingview_exchange(symbol: str) -> str:
    """
    Get tvDatafeed exchange format from symbol exchange suffix.
    
    Args:
        symbol: Stock symbol with exchange suffix (e.g., "AAPL.US", "AIR.PA")
    
    Returns:
        tvDatafeed exchange name (e.g., "NASDAQ", "EURONEXT")
    """
    upper = (symbol or "").strip().upper()
    suffix = upper.split(".")[-1] if "." in upper else ""
    config = EXCHANGE_CONFIG.get(suffix)
    if config:
        return config["tv_id"]
    return suffix


def fetch_tradingview_prices(
    symbol: str,
    max_days: int = 365
) -> Tuple[bool, str]:
    """
    Fetch historical price data from TradingView and sync to Supabase.
    Incremental: If cloud data exists, fetch only new bars.
    
    Args:
        symbol: Stock symbol with exchange suffix (e.g., "AAPL.US", "AIR.PA")
        max_days: Max historical bars to fetch if no cloud data exists
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError:
        return False, "tvDatafeed library not installed. Run: pip install tvDatafeed"
    
    import datetime as dt
    from api.stock_ai import _last_trading_day, sync_df_to_supabase, _get_supabase_info

    # Parse symbol
    upper = symbol.strip().upper()
    parts = upper.split(".")
    if len(parts) < 2:
        return False, f"Invalid symbol format: {symbol}. Expected format: SYMBOL.EXCHANGE"
    
    base_symbol = parts[0]
    exchange_suffix = parts[1]
    
    # Get tvDatafeed exchange format
    tv_exchange = get_tradingview_exchange(symbol)
    
    today = dt.date.today()
    info = _get_supabase_info(upper)
    last_date = info["last_date"]
    current_count = info["count"]
    
    is_up_to_date = last_date and last_date >= _last_trading_day(today)
    has_enough_history = current_count >= max_days
    
    if is_up_to_date and has_enough_history:
        return True, "Already up to date and sufficient history in Cloud"
    
    # Throttle slightly to avoid hammering TradingView / remote host when
    # updating many symbols in a loop.
    try:
        delay = float(os.getenv("TRADINGVIEW_REQUEST_DELAY", "1.5"))
        if delay > 0:
            time.sleep(delay)
    except Exception:
        pass

    try:
        # Initialize TvDatafeed
        tv = TvDatafeed()
        
        # Calculate how many bars we need
        # If we don't have enough history, we need max_days (plus some buffer)
        # If we are just stale, we need at least the gap since last_date
        
        needed_for_history = max_days + 100 if not has_enough_history else 0
        needed_for_update = (today - last_date).days + 10 if last_date else max_days + 30
        
        n_bars = max(needed_for_history, needed_for_update)
        # Cap at a reasonable limit (e.g., 5000) if needed, but max_days is usually smaller
        n_bars = min(5000, n_bars)

        print(f"TV FETCH: {upper} | n_bars={n_bars} (history={not has_enough_history}, stale={not is_up_to_date})")

        # Fetch historical data
        df = tv.get_hist(
            symbol=base_symbol,
            exchange=tv_exchange,
            interval=Interval.in_daily,
            n_bars=n_bars
        )
        
        if df is None or df.empty:
            return False, f"No data found for {symbol} on {tv_exchange}"
        
        # Prepare data
        df_new = df.reset_index()
        # TV returns 'datetime' column
        df_new = df_new.rename(columns={'datetime': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
        df_new['adjusted_close'] = df_new['close']
        
        # Sync Directly
        ok, sync_msg = sync_df_to_supabase(upper, df_new)
        return ok, f"OK (tradingview) - {sync_msg}"
        
    except Exception as e:
        error_msg = str(e)
        if "symbol not found" in error_msg.lower():
            return False, f"Symbol {base_symbol} not found on {tv_exchange}"
        elif "invalid exchange" in error_msg.lower():
            return False, f"Invalid exchange: {tv_exchange}"
        else:
            return False, f"TradingView error: {error_msg}"



def fetch_tradingview_fundamentals_bulk(
    tickers: List[str]
) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Bulk fetch fundamentals from TradingView screener.
    
    Args:
        tickers: List of stock symbols with exchange suffix
    
    Returns:
        Dict mapping ticker -> (data_dict, meta_dict)
    """
    if not tickers:
        return {}
    
    try:
        from tradingview_screener import Query
        try:
            from tradingview_screener import Column
        except ImportError:
            Column = None
        try:
            from tradingview_screener import col
        except ImportError:
            col = None
    except ImportError:
        return {}
    
    # Import helper functions
    from api.stock_ai import _finite_float, sync_data_to_supabase
    
    bulk_chunk_size = int(os.getenv("TRADINGVIEW_BULK_CHUNK_SIZE", "500"))
    bulk_chunk_size = max(50, min(2000, bulk_chunk_size))
    now_ts = int(time.time())
    
    # Group by TradingView market
    market_groups: Dict[str, List[str]] = defaultdict(list)
    base_to_tickers_by_market: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    # Aliases for known mismatches (Local -> TradingView)
    TV_SYMBOL_ALIASES = {
        "AIND": "ADPC",  # Arab Dairy
        "AIND.EGX": "ADPC",
        "EKHOA": "EKHO", # Egyptian Kuwaiti Holding (USD/EGP variant)
        "EKHOA.EGX": "EKHO",
        "AIVCB": "AIFI", # Atlas Investment
        "AIVCB.EGX": "AIFI",
        "ODHN": "ODIN",  # Odin Investments
        "ODHN.EGX": "ODIN",
        # Add more as discovered
    }

    for sym in tickers:
        up = (sym or "").strip().upper()
        if not up:
            continue
        
        # We try TradingView for EGX symbols as fallback if Mubasher is failing
        # (TradingView has many EGX stocks now)
        
        # Check alias first
        alias_target = TV_SYMBOL_ALIASES.get(up)
        if not alias_target:
             # Try without suffix
             base_only = up.split(".")[0]
             if base_only in TV_SYMBOL_ALIASES:
                 alias_target = TV_SYMBOL_ALIASES[base_only]

        if alias_target:
             base = alias_target
             # We need to know which market the ALIAS belongs to. 
             # For now assume same market as original derived, or just EGX if it was EGX
             # Usually aliases are within same market.
             market = get_tradingview_market(up) 
        else:
             base = up.split(".")[0]
             market = get_tradingview_market(up)
        
        market_groups[market].append(base)
        
        # KEY: Map the TRADINGVIEW BASE back to the ORIGINAL FULL SYMBOL
        # So when we get result for "ADPC", we store it under "AIND.EGX"
        base_to_tickers_by_market[market][base].append(sym)
    
    out: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    
    def _chunks(items: List[str], size: int) -> List[List[str]]:
        if size <= 0:
            return [items]
        return [items[i : i + size] for i in range(0, len(items), size)]
    
    def _has_core_fund_metrics(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        core = ["marketCap", "peRatio", "eps", "dividendYield", "beta", "high52", "low52"]
        for k in core:
            if d.get(k) is not None:
                return True
        return False
    
    # Fetch data for each market
    for market, bases in market_groups.items():
        uniq_bases = list(dict.fromkeys(bases))
        
        for chunk in _chunks(uniq_bases, bulk_chunk_size):
            try:
                q = (
                    Query()
                    .set_markets(market)
                    .select(
                        "name",
                        "description",
                        "market_cap_basic",
                        "price_earnings_ttm",
                        "earnings_per_share_basic_ttm",
                        "dividend_yield_recent",
                        "sector",
                        "industry",
                        "logoid",
                    )
                )
                
                # Apply filter based on available import
                if Column is not None:
                    q = q.where(Column("name").isin(chunk))
                elif col is not None:
                    q = q.where(col("name").isin(chunk))
                else:
                    continue
                
                _, df = q.limit(len(chunk)).get_scanner_data()
                if df is None or df.empty:
                    continue
                
                # Process each row
                for _, row in df.iterrows():
                    base = str(row.get("name") or "").strip().upper()
                    if not base:
                        continue
                    
                    # Get market cap (prefer market_cap_basic, fallback to fund_total_assets)
                    mcap = _finite_float(row.get("market_cap_basic"))
                    if mcap is None:
                        mcap = _finite_float(row.get("fund_total_assets"))
                    
                    data = {
                        "marketCap": mcap,
                        "peRatio": _finite_float(row.get("price_earnings_ttm")),
                        "eps": _finite_float(row.get("earnings_per_share_basic_ttm")),
                        "dividendYield": _finite_float(row.get("dividend_yield_recent")),
                        "sector": row.get("sector"),
                        "industry": row.get("industry"),
                        "name": row.get("description") or row.get("name"),
                        "logoUrl": f"https://s3-symbol-logo.tradingview.com/{row['logoid']}.svg" if row.get("logoid") else None
                    }
                    
                    # Skip if no core metrics
                    if not _has_core_fund_metrics(data):
                        continue
                    
                    # Sync Directly for all matching symbols
                    for full_sym in base_to_tickers_by_market[market].get(base, []):
                        sync_data_to_supabase(full_sym, data)
                        out[full_sym] = (
                            data,
                            {
                                "fetchedAt": now_ts,
                                "source": "tradingview",
                                "servedFrom": "live_tradingview_bulk",
                                "market": market,
                            },
                        )
            
            except Exception as e:
                # Log error but continue with other chunks
                print(f"Error fetching TradingView fundamentals for market {market}: {e}")
                continue
    
    return out
