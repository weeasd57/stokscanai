from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from api.live_bot import bot_instance
from api.stock_ai import get_cached_tickers

router = APIRouter(prefix="/bot", tags=["Live Bot"])

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
