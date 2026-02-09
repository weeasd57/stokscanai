from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
import os
from pathlib import Path
from datetime import datetime

from api.live_bot import bot_instance
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

@router.post("/start")
def start_bot():
    try:
        bot_instance.start()
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/stop")
def stop_bot():
    try:
        bot_instance.stop()
        return {"status": "stopping"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status")
def get_bot_status():
    try:
        return bot_instance.get_status()
    except Exception as e:
        import traceback
        print(f"Error in get_bot_status: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config")
def update_bot_config(config: BotConfigUpdate):
    try:
        updates = config.dict(exclude_unset=True)
        print(f"DEBUG: Received config update: {updates}")
        bot_instance.update_config(updates)
        status = bot_instance.get_status()
        return {"status": "updated", "config": status["config"]}
    except Exception as e:
        import traceback
        print(f"Error in update_bot_config: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available_coins")
def get_available_coins():
    """Fetches available coins from Supabase stock_ai cache."""
    try:
        tickers = get_cached_tickers()
        # Convert set of (symbol, exchange) to formatted strings
        # Only include CRYPTO or format appropriately? 
        # User asked for 'my symbols from supabase'.
        # We can try to be smart: if exchange is CRYPTO, append /USD if missing
        
        out = []
        for symbol, exchange in tickers:
            s_up = symbol.upper()
            e_up = exchange.upper()
            
            # Simple heuristic matching the bot's expected format (e.g. BTC/USD)
            if e_up in ["CRYPTO", "US", "CCC"] or "/" in s_up:
                # If it already has /, keep it
                if "/" in s_up:
                    out.append(s_up)
                elif e_up == "CRYPTO":
                    out.append(f"{s_up}/USD")
                else:
                    out.append(s_up) # e.g. Stock ticker
            else:
                 out.append(s_up)
                 
        return sorted(list(set(out)))
    except Exception as e:
        print(f"Error fetching coins: {e}")
        return []

@router.get("/performance")
def get_bot_performance():
    """تحليل شامل لأداء البوت من ملفات السجلات"""
    try:
        base_dir = Path(os.getcwd())
        logs_dir = base_dir / "logs"
        state_dir = base_dir / "state"
        
        # قراءة ملفات السجلات
        trades_file = logs_dir / "trades.json"
        performance_file = logs_dir / "performance.json"
        alerts_file = logs_dir / "alerts.json"
        state_file = state_dir / "bot_state.json"
        
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
            "summary": {
                "total_trades": len(buys),
                "completed_trades": len(sells),
                "open_positions_count": len(open_positions),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": (len(wins) / len(sells) * 100) if sells else 0,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            },
            "daily": daily_stats,
            "exit_reasons": exit_reasons,
            "symbols": symbols_stats,
            "open_positions": open_positions,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "alerts": {
                "total": len(alerts),
                "recent": recent_alerts,
                "by_type": alerts_by_type,
            },
            "last_updated": datetime.now().isoformat(),
        }
    except Exception as e:
        import traceback
        print(f"Error in get_bot_performance: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
