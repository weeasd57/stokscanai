from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
import os
from pathlib import Path
from datetime import datetime

from api.live_bot import bot_manager
from api.stock_ai import get_cached_tickers, _supabase_read_with_retry, supabase, _init_supabase

router = APIRouter(tags=["AI_BOT"])

class BotConfigUpdate(BaseModel):
    alpaca_key_id: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: Optional[str] = None
    coins: Optional[List[str]] = None
    king_threshold: Optional[float] = None
    council_threshold: Optional[float] = None
    max_notional_usd: Optional[float] = None
    pct_cash_per_trade: Optional[float] = None
    bars_limit: Optional[int] = None
    poll_seconds: Optional[int] = None
    timeframe: Optional[str] = None
    use_council: Optional[bool] = None
    data_source: Optional[str] = None
    enable_sells: Optional[bool] = None
    target_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    hold_max_bars: Optional[int] = None
    use_trailing: Optional[bool] = None
    trail_be_pct: Optional[float] = None
    trail_lock_trigger_pct: Optional[float] = None
    trail_lock_pct: Optional[float] = None
    save_to_supabase: Optional[bool] = None
    king_model_path: Optional[str] = None
    council_model_path: Optional[str] = None
    max_open_positions: Optional[int] = None
    name: Optional[str] = None
    execution_mode: Optional[str] = None
    trading_mode: Optional[str] = None

class BotCreate(BaseModel):
    bot_id: str
    name: str

@router.post("/start")
def start_bot(bot_id: str = "primary"):
    try:
        bot = bot_manager.get_bot(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
        bot.start()
        return {"status": "started", "bot_id": bot_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/stop")
def stop_bot(bot_id: str = "primary"):
    try:
        bot = bot_manager.get_bot(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
        bot.stop()
        return {"status": "stopping", "bot_id": bot_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status")
def get_bot_status(bot_id: str = "primary"):
    try:
        bot = bot_manager.get_bot(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
        return bot.get_status()
    except Exception as e:
        import traceback
        print(f"Error in get_bot_status: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config")
def update_bot_config(config: BotConfigUpdate, bot_id: str = "primary"):
    try:
        bot = bot_manager.get_bot(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
        
        updates = config.dict(exclude_unset=True)
        print(f"DEBUG: Received config update for {bot_id}: {updates}")
        bot.update_config(updates)
        bot_manager.save_bots()
        status = bot.get_status()
        return {"status": "updated", "config": status["config"]}
    except Exception as e:
        import traceback
        print(f"Error in update_bot_config: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/countries")
def get_available_countries():
    """Returns a list of available countries and their symbol counts from summary files."""
    try:
        base_dir = Path(os.getcwd())
        symbols_dir = base_dir / "symbols_data"
        
        if not symbols_dir.exists():
            return []
            
        # Try to find country_summary_*.json
        summary_files = list(symbols_dir.glob("country_summary_*.json"))
        if summary_files:
            # Take the newest one
            newest = max(summary_files, key=os.path.getmtime)
            with open(newest, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Return list of objects with name and count
                result = []
                for country_name, info in data.items():
                    count = info.get("TotalSymbols", 0)
                    if count > 0:
                        result.append({"name": country_name, "count": count})
                return sorted(result, key=lambda x: x["name"])
        
        # Fallback: Parse filenames if summary is missing
        country_files = list(symbols_dir.glob("*_all_symbols_*.json"))
        countries = set()
        for f in country_files:
            name = f.name.split("_all_symbols_")[0]
            if name and name != "all":
                countries.add(name)
        return [{"name": c, "count": 0} for c in sorted(list(countries))]
    except Exception as e:
        print(f"Error fetching countries: {e}")
        return []

@router.get("/models")
def get_available_models():
    """Returns a list of model files (.pkl) from the api/models directory."""
    try:
        base_dir = Path(os.getcwd())
        models_dir = base_dir / "api" / "models"
        
        if not models_dir.exists():
            return []
            
        # Find all .pkl files
        model_files = list(models_dir.glob("*.pkl"))
        # Return paths relative to models_dir or just the names? 
        # The bot seems to expect a path. Let's return the relative path from base_dir for now.
        return sorted([str(f.relative_to(base_dir)).replace("\\", "/") for f in model_files])
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

@router.get("/available_coins")
def get_available_coins(source: Optional[str] = None, limit: int = 0, country: Optional[str] = None, pair_type: Optional[str] = None):
    """Fetches available coins from various sources including local files."""
    try:
        if source == "alpaca_stocks":
            base_dir = Path(os.getcwd())
            alpaca_json = base_dir / "alpaca_exchanges" / "us_equity" / "all_assets.json"
            if alpaca_json.exists():
                with open(alpaca_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    symbols = [item.get("symbol") for item in data if item.get("symbol") and item.get("status") == "active"]
                    return sorted(list(set(symbols))) if limit <= 0 else sorted(list(set(symbols)))[:limit]
            return []

        if source == "global" and country:
            base_dir = Path(os.getcwd())
            symbols_dir = base_dir / "symbols_data"
            
            # Find the file for this country - be case-insensitive
            country_files = list(symbols_dir.glob(f"{country}_all_symbols_*.json"))
            if not country_files:
                # Try partial match or alternate case if exactly country name glob fails
                all_files = list(symbols_dir.glob("*_all_symbols_*.json"))
                country_files = [f for f in all_files if f.name.lower().startswith(country.lower())]

            if country_files:
                # Take the newest
                newest = max(country_files, key=os.path.getmtime)
                with open(newest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Some files use "Symbol", some use "Code"
                    symbols = [item.get("Symbol") or item.get("Code") for item in data if item.get("Symbol") or item.get("Code")]
                    
                    # Ensure limit=0 returns all
                    effective_limit = len(symbols) if limit <= 0 else limit
                    return sorted(list(set(symbols)))[:effective_limit]
            return []

        if source == "binance":
            from api.binance_data import fetch_all_binance_symbols
            return fetch_all_binance_symbols(quote_asset="USDT", limit=limit)
        
        if source == "alpaca":
            # Load from local alpaca_exchanges/crypto/CRYPTO.json
            base_dir = Path(os.getcwd())
            alpaca_json = base_dir / "alpaca_exchanges" / "crypto" / "CRYPTO.json"
            
            # Additional check: look relative to this file's directory
            if not alpaca_json.exists():
                alpaca_json = Path(__file__).parent.parent / "alpaca_exchanges" / "crypto" / "CRYPTO.json"

            if alpaca_json.exists():
                with open(alpaca_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Filter for symbols with status="active" and return as strings
                    symbols = [item.get("symbol") for item in data if item.get("status") == "active"]
                    # Apply pair_type filter (e.g. 'USD' -> only /USD pairs)
                    if pair_type and pair_type != "ALL":
                        symbols = [s for s in symbols if s and s.endswith(f"/{pair_type}")]
                    return sorted(list(set(symbols))) if limit <= 0 else sorted(list(set(symbols)))[:limit]
            
            # Fallback for remote deployments (Hugging Face / Vercel)
            # We prioritize /USD pairs as most accounts have USD cash, not USDC/USDT.
            usd_pairs = [
                "AAVE/USD", "AVAX/USD", "BAT/USD", "BCH/USD", "BTC/USD", "CRV/USD",
                "DOGE/USD", "DOT/USD", "ETH/USD", "GRT/USD", "LINK/USD", "LTC/USD",
                "PEPE/USD", "SHIB/USD", "SOL/USD", "SUSHI/USD", "TRUMP/USD", "UNI/USD",
                "USDC/USD", "USDT/USD", "XRP/USD", "XTZ/USD", "YFI/USD"
            ]
            stable_pairs = [
                "AAVE/USDC", "AAVE/USDT", "AVAX/USDC", "AVAX/USDT", "BAT/USDC",
                "BCH/BTC", "BCH/USDC", "BCH/USDT", "BTC/USDC", "BTC/USDT", "CRV/USDC",
                "DOGE/USDC", "DOGE/USDT", "DOT/USDC", "ETH/BTC", "ETH/USDC", "ETH/USDT",
                "GRT/USDC", "LINK/BTC", "LINK/USDC", "LINK/USDT", "LTC/BTC", "LTC/USDC",
                "LTC/USDT", "SHIB/USDC", "SHIB/USDT", "SOL/USDC", "SOL/USDT",
                "SUSHI/USDC", "SUSHI/USDT", "UNI/BTC", "UNI/USDC", "UNI/USDT",
                "USDT/USDC", "XTZ/USDC", "YFI/USDC", "YFI/USDT"
            ]
            fallback = usd_pairs + stable_pairs
            # Apply pair_type filter
            if pair_type and pair_type != "ALL":
                fallback = [s for s in fallback if s.endswith(f"/{pair_type}")]
            return sorted(fallback)[:limit] if limit > 0 else sorted(fallback)
            
        tickers = get_cached_tickers()
        # Convert set of (symbol, exchange) to formatted strings
        out = []
        for symbol, exchange in tickers:
            s_up = symbol.upper()
            e_up = exchange.upper()
            
            if e_up in ["CRYPTO", "US", "CCC"] or "/" in s_up:
                if "/" in s_up:
                    out.append(s_up)
                elif e_up == "CRYPTO":
                    out.append(f"{s_up}/USD")
                else:
                    out.append(s_up)
            else:
                 out.append(s_up)
                 
        return sorted(list(set(out)))
    except Exception as e:
        print(f"Error fetching coins: {e}")
        return []

from pydantic import BaseModel
class BotCreate(BaseModel):
    bot_id: str
    name: str
    alpaca_key_id: Optional[str] = None
    alpaca_secret_key: Optional[str] = None

@router.get("/list")
def list_bots():
    return {"bots": bot_manager.list_bots()}

@router.post("/create")
def create_bot(req: BotCreate):
    try:
        bot_manager.create_bot(
            req.bot_id, 
            req.name,
            alpaca_key_id=req.alpaca_key_id,
            alpaca_secret_key=req.alpaca_secret_key
        )
        return {"status": "created", "bot_id": req.bot_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/delete/{bot_id}")
def delete_bot(bot_id: str):
    try:
        bot_manager.delete_bot(bot_id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/alpaca_watchlist")
def get_alpaca_watchlist(bot_id: str = "primary"):
    """Fetches symbols from the user's Alpaca primary watchlist."""
    from api.alpaca_adapter import create_alpaca_client
    try:
        bot = bot_manager.get_bot(bot_id)
        if not bot:
             raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
        cfg = bot.config
        if not cfg.alpaca_key_id or not cfg.alpaca_secret_key:
             raise HTTPException(status_code=400, detail="Alpaca keys not configured")
             
        client = create_alpaca_client(
            key_id=cfg.alpaca_key_id,
            secret_key=cfg.alpaca_secret_key,
            base_url=cfg.alpaca_base_url
        )
        
        # Try to find 'KING' or 'Primary' or just take the first one
        watchlists = client.get_watchlists()
        if not watchlists:
             return []
             
        target_wl = None
        for wl in watchlists:
             if wl.name.upper() in ["KING", "PRIMARY", "AI_BOT"]:
                  target_wl = client.get_watchlist_by_id(wl.id)
                  break
        
        if not target_wl:
             target_wl = client.get_watchlist_by_id(watchlists[0].id)
             
        symbols = []
        if target_wl and hasattr(target_wl, "assets"):
             for asset in target_wl.assets:
                  sym = asset["symbol"]
                  # Heuristic: if it's crypto but missing USD, append it
                  # Actually let's just return what's there
                  symbols.append(sym)
                  
        return sorted(list(set(symbols)))
    except Exception as e:
        print(f"Error fetching Alpaca watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
def get_bot_performance(bot_id: str = "primary"):
    """Comprehensive analysis of bot performance from log files"""
    try:
        # Try to fetch from Supabase first
        _init_supabase()
        if supabase:
            try:
                # Query bot_trades table for this bot
                response = supabase.table("bot_trades").select("*").eq("bot_id", bot_id).order("timestamp", desc=True).execute()
                trades = response.data or []
                
                buys = [t for t in trades if t.get("action") == "BUY"]
                sells = [t for t in trades if t.get("action") == "SELL"]
                
                wins = [t for t in sells if (t.get("pnl") or 0) > 0]
                losses = [t for t in sells if (t.get("pnl") or 0) <= 0]
                
                total_pnl = sum((t.get("pnl") or 0) for t in sells)
                
                # Exit reasons from Supabase metadata or action
                exit_reasons = {}
                for trade in sells:
                    reason = trade.get("metadata", {}).get("reason") or "Manual/Unknown"
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                
                # Symbol stats
                symbols_stats = {}
                for trade in sells:
                    symbol = trade.get("symbol", "Unknown")
                    if symbol not in symbols_stats:
                        symbols_stats[symbol] = {"count": 0, "pnl": 0, "wins": 0}
                    symbols_stats[symbol]["count"] += 1
                    symbols_stats[symbol]["pnl"] += (trade.get("pnl") or 0)
                    if (trade.get("pnl") or 0) > 0:
                        symbols_stats[symbol]["wins"] += 1
                
                for s in symbols_stats:
                    symbols_stats[s]["win_rate"] = (symbols_stats[s]["wins"] / symbols_stats[s]["count"] * 100) if symbols_stats[s]["count"] > 0 else 0

                # Open positions from bot state (real-time)
                bot = bot_manager.get_bot(bot_id)
                open_positions = []
                if bot:
                    for s, pos_data in bot._pos_state.items():
                         open_positions.append({
                            "symbol": s,
                            "entry_price": pos_data.get("entry_price"),
                            "entry_time": pos_data.get("entry_time"),
                            "bars_held": pos_data.get("bars_held", 0),
                            "trail_mode": pos_data.get("trail_mode", "NONE"),
                        })

                return {
                    "total_trades": len(buys),
                    "win_rate": (len(wins) / len(sells) * 100) if sells else 0,
                    "profit_loss": total_pnl,
                    "profit_loss_pct": 0, # Could be calculated if balance is tracked in DB
                    "avg_trade_profit": (total_pnl / len(sells)) if sells else 0,
                    "exit_reasons": exit_reasons,
                    "symbol_performance": symbols_stats,
                    "open_positions": open_positions,
                    "trades": trades,
                    "last_updated": datetime.now().isoformat(),
                    "source": "supabase"
                }
            except Exception as e:
                print(f"Supabase performance fetch error: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to local files below...

        base_dir = Path(os.getcwd())
        logs_dir = base_dir / "logs"
        state_dir = base_dir / "state"
        
        # Read log files relevant to this bot
        trades_file = logs_dir / f"{bot_id}_trades.json"
        performance_file = logs_dir / f"{bot_id}_performance.json"
        alerts_file = logs_dir / "alerts.json" # تنبيهات عامة أو يمكن تخصيصها لاحقاً
        state_file = state_dir / "bot_state.json" # قديم، BotConfig يخزن الحالة الآن
        
        def load_json_file(filepath):
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except:
                    return {}
            return {}
        
        trades_data = load_json_file(trades_file)
        performance_data = load_json_file(performance_file)
        alerts_data = load_json_file(alerts_file)
        state_data = load_json_file(state_file)
        
        # Trade analysis
        trades = trades_data.get("trades", [])
        buys = [t for t in trades if t.get("action") == "BUY"]
        sells = [t for t in trades if t.get("action") == "SELL"]
        
        wins = [t for t in sells if t.get("pnl", 0) > 0]
        losses = [t for t in sells if t.get("pnl", 0) <= 0]
        
        total_pnl = sum(t.get("pnl", 0) for t in sells)
        avg_win = sum(t.get("pnl", 0) for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.get("pnl", 0) for t in losses) / len(losses) if losses else 0
        
        # Exit reasons
        exit_reasons = {}
        for trade in sells:
            reason = trade.get("reason", "Unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Trades by symbol
        symbols_stats = {}
        for trade in sells:
            symbol = trade.get("symbol", "Unknown")
            if symbol not in symbols_stats:
                symbols_stats[symbol] = {"count": 0, "pnl": 0, "wins": 0}
            symbols_stats[symbol]["count"] += 1
            symbols_stats[symbol]["pnl"] += trade.get("pnl", 0)
            if trade.get("pnl", 0) > 0:
                symbols_stats[symbol]["wins"] += 1
        
        # Calculate win rate for each symbol
        for symbol in symbols_stats:
            count = symbols_stats[symbol]["count"]
            symbols_stats[symbol]["win_rate"] = (symbols_stats[symbol]["wins"] / count * 100) if count > 0 else 0
        
        # Daily performance
        daily_stats = {
            "date": performance_data.get("date", datetime.now().strftime("%Y-%m-%d")),
            "trades_count": performance_data.get("trades_count", 0),
            "wins": performance_data.get("wins", 0),
            "losses": performance_data.get("losses", 0),
            "total_pnl": performance_data.get("total_pnl", 0),
            "starting_balance": performance_data.get("starting_balance", 0),
            "current_balance": performance_data.get("current_balance", 0),
        }
        
        if daily_stats["starting_balance"] > 0:
            daily_stats["daily_return_pct"] = ((daily_stats["current_balance"] - daily_stats["starting_balance"]) / daily_stats["starting_balance"] * 100)
        else:
            daily_stats["daily_return_pct"] = 0
        
        # Alerts
        alerts = alerts_data.get("alerts", [])
        recent_alerts = alerts[-10:] if len(alerts) > 10 else alerts
        
        alerts_by_type = {}
        for alert in alerts:
            alert_type = alert.get("type", "Unknown")
            alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1
        
        # Open positions
        open_positions = []
        for symbol, pos_data in state_data.items():
            if isinstance(pos_data, dict) and "entry_price" in pos_data:
                open_positions.append({
                    "symbol": symbol,
                    "entry_price": pos_data.get("entry_price"),
                    "entry_time": pos_data.get("entry_time"),
                    "bars_held": pos_data.get("bars_held", 0),
                    "trail_mode": pos_data.get("trail_mode", "NONE"),
                })
        
        # Best and worst trade
        best_trade = max(sells, key=lambda x: x.get("pnl", 0)) if sells else None
        worst_trade = min(sells, key=lambda x: x.get("pnl", 0)) if sells else None
        
        return {
            "total_trades": len(buys),
            "win_rate": (len(wins) / len(sells) * 100) if sells else 0,
            "profit_loss": total_pnl,
            "profit_loss_pct": daily_stats.get("daily_return_pct", 0),
            "avg_trade_profit": (total_pnl / len(sells)) if sells else 0,
            "exit_reasons": exit_reasons,
            "symbol_performance": symbols_stats,
            "open_positions": open_positions,
            "trades": trades[::-1], # Reversed to show newest first
            "last_updated": datetime.now().isoformat(),
        }
    except Exception as e:
        import traceback
        print(f"Error in get_bot_performance: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
def get_bot_logs(bot_id: str = "primary", lines: int = 100):
    """Returns the last N lines of logs from the running bot."""
    try:
        bot = bot_manager.get_bot(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
        
        # Get logs from deque (last N)
        all_logs = list(bot._logs)
        requested_logs = all_logs[-lines:] if len(all_logs) > lines else all_logs
        
        return {
            "bot_id": bot_id,
            "count": len(requested_logs),
            "logs": requested_logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/candles")
def get_candles(symbol: str, bot_id: str = "primary", limit: int = 200):
    """Fetch OHLC candle data + entry/exit markers for a symbol."""
    try:
        _init_supabase()
        if not supabase:
            raise HTTPException(status_code=503, detail="Supabase not available")

        # Try to resolve normalized symbol (e.g. BATUSD -> BAT/USD) using bot config
        if bot and hasattr(bot.config, "coins") and bot.config.coins:
            # Create map of normalized -> original
            norm_map = {c.upper().replace("/", "").replace("-", "").replace("_", ""): c for c in bot.config.coins}
            clean_sym = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
            if clean_sym in norm_map:
                symbol = norm_map[clean_sym]
                print(f"DEBUG: Resolved {clean_sym} to {symbol}")

        # The bot stores symbol as BTC (without /USD suffix) in stock_bars_intraday
        db_symbol = symbol.split("/")[0] if "/" in symbol else symbol

        # Get the bot's timeframe config so we query the right data
        bot = bot_manager.get_bot(bot_id)
        timeframe = "15Min"
        if bot:
            raw_tf = getattr(bot.config, "timeframe", "15Min") or "15Min"
            # Map common formats to DB format
            tf_map = {
                "1min": "1Min", "5min": "5Min", "15min": "15Min", "30min": "30Min",
                "1h": "1Hour", "1hour": "1Hour", "4h": "4Hour", "4hour": "4Hour",
                "1d": "1Day", "1day": "1Day",
                "1Min": "1Min", "5Min": "5Min", "15Min": "15Min", "30Min": "30Min",
                "1Hour": "1Hour", "4Hour": "4Hour", "1Day": "1Day",
            }
            timeframe = tf_map.get(raw_tf.lower(), raw_tf) if raw_tf else "15Min"

        # Determine exchange - simple heuristic: if it has /USD or /USDT it's crypto, 
        # unless it's a known stock format. In this bot, Cryptos are often /USD.
        exchange = "CRYPTO"
        if bot:
            ds = getattr(bot.config, "data_source", "").lower()
            if "alpaca" in ds and "/" not in symbol:
                exchange = "US"
        elif "/" not in symbol:
            exchange = "US"

        # Query OHLC data with retry
        MAX_RETRIES = 2
        candles_resp = None
        for attempt in range(MAX_RETRIES):
            try:
                candles_resp = supabase.table("stock_bars_intraday") \
                    .select("ts,open,high,low,close,volume") \
                    .eq("symbol", db_symbol) \
                    .eq("exchange", exchange) \
                    .eq("timeframe", timeframe) \
                    .order("ts", desc=True) \
                    .limit(limit) \
                    .execute()
                break # Success
            except Exception as e:
                print(f"DEBUG: Candle fetch attempt {attempt+1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    pass # Allow empty result fallback or bubbling
                else:
                    import time # Import time for sleep
                    time.sleep(0.5)
        
        raw_candles = candles_resp.data or [] if candles_resp else []
        # Reverse to chronological order
        raw_candles.reverse()

        # Format for lightweight-charts: time as unix timestamp
        candles = []
        for c in raw_candles:
            try:
                ts = c.get("ts", "")
                # Parse ISO timestamp to unix seconds
                if ts:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    unix_ts = int(dt.timestamp())
                else:
                    continue
                candles.append({
                    "time": unix_ts,
                    "open": float(c.get("open", 0)),
                    "high": float(c.get("high", 0)),
                    "low": float(c.get("low", 0)),
                    "close": float(c.get("close", 0)),
                    "volume": int(c.get("volume", 0) or 0),
                })
            except Exception:
                continue

        # Query entry/exit markers from bot_trades
        markers_resp = supabase.table("bot_trades") \
            .select("timestamp,action,price,entry_price,pnl") \
            .eq("bot_id", bot_id) \
            .eq("symbol", symbol) \
            .order("timestamp", desc=False) \
            .execute()

        raw_markers = markers_resp.data or []
        print(f"DEBUG: /candles for {symbol} - Found {len(raw_candles)} candles and {len(raw_markers)} markers")
        markers = []
        for m in raw_markers:
            try:
                ts = m.get("timestamp", "")
                if ts:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    unix_ts = int(dt.timestamp())
                else:
                    continue

                action = m.get("action", "").upper()
                price = float(m.get("price", 0) or 0)
                pnl = m.get("pnl")

                if action in ("BUY", "SIGNAL"):
                    markers.append({
                        "time": unix_ts,
                        "position": "belowBar",
                        "color": "#22c55e",
                        "shape": "arrowUp",
                        "text": f"BUY ${price:.2f}",
                        "price": price,
                    })
                elif action == "SELL":
                    pnl_txt = f" P&L: ${float(pnl):.2f}" if pnl is not None else ""
                    markers.append({
                        "time": unix_ts,
                        "position": "aboveBar",
                        "color": "#ef4444" if (pnl or 0) <= 0 else "#22c55e",
                        "shape": "arrowDown",
                        "text": f"SELL ${price:.2f}{pnl_txt}",
                        "price": price,
                    })
            except Exception:
                continue

        # Get current entry price if position is open
        entry_price = None
        if bot:
            from api.live_bot import _normalize_symbol
            pos = bot._pos_state.get(_normalize_symbol(symbol))
            if pos:
                entry_price = pos.get("entry_price")

        return {
            "symbol": symbol,
            "candles": candles,
            "markers": markers,
            "entry_price": entry_price,
            "timeframe": timeframe,
            "count": len(candles),
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in get_candles: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
