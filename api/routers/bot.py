from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
import os
from pathlib import Path
from datetime import datetime

from api.live_bot import bot_manager
from api.stock_ai import get_cached_tickers

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
def get_available_coins(source: Optional[str] = None, limit: int = 0, country: Optional[str] = None):
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

@router.get("/list")
def list_bots():
    return {"bots": bot_manager.list_bots()}

@router.post("/create")
def create_bot(req: BotCreate):
    try:
        bot_manager.create_bot(req.bot_id, req.name)
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
    """تحليل شامل لأداء البوت من ملفات السجلات"""
    try:
        base_dir = Path(os.getcwd())
        logs_dir = base_dir / "logs"
        state_dir = base_dir / "state"
        
        # قراءة ملفات السجلات المختصة بهذا البوت
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
        
        # تحليل الصفقات
        trades = trades_data.get("trades", [])
        buys = [t for t in trades if t.get("action") == "BUY"]
        sells = [t for t in trades if t.get("action") == "SELL"]
        
        wins = [t for t in sells if t.get("pnl", 0) > 0]
        losses = [t for t in sells if t.get("pnl", 0) <= 0]
        
        total_pnl = sum(t.get("pnl", 0) for t in sells)
        avg_win = sum(t.get("pnl", 0) for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.get("pnl", 0) for t in losses) / len(losses) if losses else 0
        
        # أسباب الخروج
        exit_reasons = {}
        for trade in sells:
            reason = trade.get("reason", "Unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # الصفقات حسب العملة
        symbols_stats = {}
        for trade in sells:
            symbol = trade.get("symbol", "Unknown")
            if symbol not in symbols_stats:
                symbols_stats[symbol] = {"count": 0, "pnl": 0, "wins": 0}
            symbols_stats[symbol]["count"] += 1
            symbols_stats[symbol]["pnl"] += trade.get("pnl", 0)
            if trade.get("pnl", 0) > 0:
                symbols_stats[symbol]["wins"] += 1
        
        # حساب win rate لكل عملة
        for symbol in symbols_stats:
            count = symbols_stats[symbol]["count"]
            symbols_stats[symbol]["win_rate"] = (symbols_stats[symbol]["wins"] / count * 100) if count > 0 else 0
        
        # الأداء اليومي
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
        
        # التنبيهات
        alerts = alerts_data.get("alerts", [])
        recent_alerts = alerts[-10:] if len(alerts) > 10 else alerts
        
        alerts_by_type = {}
        for alert in alerts:
            alert_type = alert.get("type", "Unknown")
            alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1
        
        # الصفقات المفتوحة
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
        
        # أفضل وأسوأ صفقة
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
