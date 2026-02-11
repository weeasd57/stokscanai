import os
import time
import threading
import warnings
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from collections import deque
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
try:
    from tvDatafeed import TvDatafeed, Interval
except ImportError:
    TvDatafeed = None
    Interval = None

# We'll use the existing imports from run_live_bot.py logic
# assuming api module structure is available.
from api.stock_ai import _supabase_upsert_with_retry
from api.alpaca_adapter import create_alpaca_client

warnings.filterwarnings("ignore")

@dataclass
class BotConfig:
    name: str = "Primary Bot"
    execution_mode: str = "BOTH"  # "ALPACA" | "TELEGRAM" | "BOTH"
    alpaca_key_id: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    coins: list[str] = None
    king_threshold: float = 0.60
    council_threshold: float = 0.35
    max_notional_usd: float = 500.0
    pct_cash_per_trade: float = 0.10
    bars_limit: int = 200
    poll_seconds: int = 300
    timeframe: str = "1Hour"
    use_council: bool = True
    data_source: str = "binance"

    # Risk
    max_open_positions: int = 5
    enable_sells: bool = True
    target_pct: float = 0.10
    stop_loss_pct: float = 0.05
    hold_max_bars: int = 20
    use_trailing: bool = True
    trail_be_pct: float = 0.05
    trail_lock_trigger_pct: float = 0.08
    trail_lock_pct: float = 0.05
    
    # Supabase integration
    save_to_supabase: bool = False  # Save polling data to Supabase
    
    # Model Paths
    king_model_path: str = "api/models/KING_CRYPTO 👑.pkl"
    council_model_path: str = "api/models/COUNCIL_CRYPTO.pkl"

def _read_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    v = os.getenv(name, default)
    if required and not v:
        # Don't raise here for library usage, just return None or let caller handle
        return None
    return v

def _read_first_env(names: list[str], default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default

def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)

def _parse_float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)

def _parse_coins(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return raw
    coins = []
    for part in (str(raw or "")).split(","):
        s = part.strip().upper()
        if not s:
            continue
        coins.append(s)
    return coins

def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper().replace("/", "")

def _format_symbol_for_bot(symbol: str) -> str:
    """
    Alpaca positions may return crypto like 'BTCUSD'. The bot/data sources generally use 'BTC/USD'.
    """
    s = (symbol or "").strip().upper()
    if not s:
        return s
    if "/" in s:
        return s
    if len(s) > 3 and (s.endswith("USD") or s.endswith("USDT")):
        base = s[:-3] if s.endswith("USD") else s[:-4]
        quote = "USD" if s.endswith("USD") else "USDT"
        return f"{base}/{quote}"
    return s

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        n = float(value)
        if not np.isfinite(n):
            return int(default)
        # Supabase `stock_prices.volume` is a bigint in this project; crypto volume can be fractional.
        return int(round(n))
    except Exception:
        return int(default)

def _to_intraday_timeframe(value: str) -> str:
    """
    Map UI/bot timeframe strings to the DB timeframe string (e.g. '5m', '1h', '1d').
    Accepts UI values like '5Min', '1Hour', '4Hour', as well as already-normalized values like '5m'.
    """
    import re

    s = (value or "").strip().lower()
    if not s:
        return "1h"

    # Already-normalized values
    allowed = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"}
    if s in allowed:
        return s

    m = re.match(r"^\s*(\d+)\s*([a-z]+)\s*$", s)
    if not m:
        # Back-compat: treat legacy labels without a number as 1h/min/day.
        if "day" in s:
            return "1d"
        if "week" in s:
            return "1w"
        if "min" in s:
            return "1m"
        if "hour" in s:
            return "1h"
        return "1h"

    amount = int(m.group(1))
    unit_raw = m.group(2)

    if unit_raw in {"min", "mins", "minute", "minutes", "m"}:
        out = f"{amount}m"
    elif unit_raw in {"hour", "hours", "h"}:
        out = f"{amount}h"
    elif unit_raw in {"day", "days", "d"}:
        out = f"{amount}d" if amount != 1 else "1d"
    elif unit_raw in {"week", "weeks", "w"}:
        out = f"{amount}w" if amount != 1 else "1w"
    else:
        out = "1h"

    # Normalize common equivalents
    if out == "60m":
        out = "1h"
    if out == "24h":
        out = "1d"

    return out if out in allowed else ("1m" if out.endswith("m") else ("1h" if out.endswith("h") else "1d"))

class LiveBot:
    def __init__(self, bot_id: str = "primary", config: Optional[BotConfig] = None):
        self.bot_id = bot_id
        # Re-entrant: several public methods call `_log()` while holding the lock.
        # A plain Lock would deadlock and make `/bot/status` hang forever.
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._logs = deque(maxlen=1000)
        self._trades = deque(maxlen=100)
        self._status = "stopped"  # stopped, running, error
        self._last_scan_time = None
        self._error_msg = None
        self._data_stream = {} # {symbol: {source, count, timestamp, status, error}}
        self._started_at = None  # Track when bot started
        
        # Default config from env or defaults
        self.config = config or self._build_default_config()
        
        # Models
        self.king_obj = None
        self.king_clf = None
        self.validator = None
        self.api = None
        # Per-symbol runtime state for exit logic
        self._pos_state: Dict[str, Dict[str, Any]] = {}

        # Telegram Bridge
        self.telegram_bridge = None
        
        # Persistence
        self._logs_dir = Path("logs")
        self._logs_dir.mkdir(exist_ok=True)
        self._trades_file = self._logs_dir / f"{self.bot_id}_trades.json"
        self._perf_file = self._logs_dir / f"{self.bot_id}_performance.json"
        self._load_persistent_data()

    def _load_persistent_data(self):
        """Loads trades and stats from JSON files."""
        if self._trades_file.exists() and self._trades_file.stat().st_size > 0:
            try:
                with open(self._trades_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    trades = data.get("trades", [])
                    self._trades = deque(trades, maxlen=1000)
            except Exception as e:
                print(f"Error loading trades: {e}")

    def _save_trade_persistent(self, trade_info: Dict[str, Any]):
        """Append a trade to the persistent JSON file."""
        try:
            trades = []
            if self._trades_file.exists() and self._trades_file.stat().st_size > 0:
                with open(self._trades_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        trades = data.get("trades", [])
                    except json.JSONDecodeError:
                        trades = []
            
            trades.append(trade_info)
            # Keep last 5000 trades in file
            if len(trades) > 5000:
                trades = trades[-5000:]
                
            with open(self._trades_file, "w", encoding="utf-8") as f:
                json.dump({"trades": trades, "last_updated": datetime.now().isoformat()}, f, indent=2)
                
            # Update performance.json
            self._update_performance_stats(trades)
        except Exception as e:
            self._log(f"Persistence Error: {e}")

    def _update_performance_stats(self, trades: List[Dict[str, Any]]):
        """Calculate and save cumulative stats."""
        try:
            sells = [t for t in trades if t.get("action") == "SELL"]
            wins = [t for t in sells if t.get("pnl", 0) > 0]
            total_pnl = sum(t.get("pnl", 0) for t in sells)
            
            perf = {
                "total_trades": len(trades),
                "completed_trades": len(sells),
                "wins": len(wins),
                "losses": len(sells) - len(wins),
                "win_rate": (len(wins) / len(sells) * 100) if sells else 0,
                "total_pnl": total_pnl,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self._perf_file, "w", encoding="utf-8") as f:
                json.dump(perf, f, indent=2)
        except Exception as e:
            print(f"Stats Error: {e}")

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{self.config.name}] [{ts}] {msg}"
        with self._lock:
            print(line) # Keep printing to stdout for convenience
            self._logs.append(line)
            
        # Optional: Send important logs to Telegram
        if self.telegram_bridge and self.config.execution_mode in ["TELEGRAM", "BOTH"]:
            # Only send orders and critical errors here. Signals handled separately.
            if "order" in msg.lower() or "CRITICAL" in msg:
                 self.telegram_bridge.send_notification(f"ℹ️ *{self.config.name}*\n{msg}")

    def set_telegram_bridge(self, bridge):
        self.telegram_bridge = bridge

    def _send_telegram_signal(self, symbol: str, price: float, notional: float, king_conf: float, council_conf: Optional[float] = None):
        """Send a professional looking trade signal to Telegram."""
        if not self.telegram_bridge or self.config.execution_mode not in ["TELEGRAM", "BOTH"]:
            return
            
        target_price = price * (1 + self.config.target_pct)
        stop_price = price * (1 - self.config.stop_loss_pct)
        
        mode_emoji = "🔵" if self.config.execution_mode == "TELEGRAM" else "🟢"
        title = f"{mode_emoji} *NEW TRADE SIGNAL*"
        if self.config.execution_mode == "TELEGRAM":
            title += " (Signal Only)"
        
        msg = (
            f"{title}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💎 *Symbol:* `{symbol}`\n"
            f"💰 *Entry Price:* `${price:,.4f}`\n"
            f"📈 *Target (TP):* `${target_price:,.4f}` (+{self.config.target_pct*100:.1f}%)\n"
            f"📉 *Stop Loss (SL):* `${stop_price:,.4f}` (-{self.config.stop_loss_pct*100:.1f}%)\n"
            f"💵 *Amount:* `${notional:,.2f}`\n"
            f"━━━━━━━━━━━━━━━\n"
            f"👑 *KING Conf:* `{king_conf:.2f}`\n"
        )
        if council_conf is not None:
            msg += f"🛡️ *COUNCIL Conf:* `{council_conf:.2f}`\n"
            
        msg += f"🤖 *Bot:* `{self.config.name}`"
        
        self.telegram_bridge.send_notification(msg)

    def _save_signal_record(self, symbol: str, price: float, notional: float, king_conf: float, council_conf: Optional[float] = None, action: str = "BUY"):
        """Save a signal as a virtual trade in the history so it shows up in UI."""
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "amount": float(notional),
            "price": float(price),
            "king_conf": float(king_conf),
            "council_conf": float(council_conf) if council_conf is not None else None,
            "order_id": "signal_only"
        }
        self._trades.append(trade)
        self._save_trade_persistent(trade)
        self._log(f"{action} SIGNAL RECORDED: {symbol} @ {price:.4f}")

    def _build_default_config(self) -> BotConfig:
        coins = _parse_coins(
            _read_env(
                "LIVE_COINS",
                "BTC/USD,ETH/USD,SOL/USD,LTC/USD,LINK/USD,DOGE/USD,AVAX/USD,MATIC/USD",
            )
        )
        return BotConfig(
            # Support both env var styles used across the repo/UI.
            alpaca_key_id=str(_read_first_env(["ALPACA_KEY_ID", "ALPACA_API_KEY"], "") or ""),
            alpaca_secret_key=str(_read_first_env(["ALPACA_SECRET_KEY"], "") or ""),
            alpaca_base_url=str(_read_env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")),
            coins=coins,
            king_threshold=_parse_float(_read_env("KING_THRESHOLD", "0.60"), 0.60),
            council_threshold=_parse_float(_read_env("COUNCIL_THRESHOLD", "0.35"), 0.35),
            max_notional_usd=_parse_float(_read_env("MAX_NOTIONAL_USD", "500"), 500.0),
            pct_cash_per_trade=_parse_float(_read_env("PCT_CASH_PER_TRADE", "0.10"), 0.10),
            bars_limit=int(float(_read_env("BARS_LIMIT", "200") or 200)),
            poll_seconds=int(float(_read_env("POLL_SECONDS", "300") or 300)),
            timeframe=str(_read_env("TIMEFRAME", "1Hour")),
            data_source=str(_read_env("LIVE_DATA_SOURCE", "binance") or "binance").strip().lower(),
            enable_sells=_parse_bool(_read_env("LIVE_ENABLE_SELLS", "1"), True),
            target_pct=_parse_float(_read_env("LIVE_TARGET_PCT", "0.10"), 0.10),
            stop_loss_pct=_parse_float(_read_env("LIVE_STOP_LOSS_PCT", "0.05"), 0.05),
            hold_max_bars=int(float(_read_env("LIVE_HOLD_MAX_BARS", "20") or 20)),
            use_trailing=_parse_bool(_read_env("LIVE_USE_TRAILING", "1"), True),
            trail_be_pct=_parse_float(_read_env("LIVE_TRAIL_BE_PCT", "0.05"), 0.05),
            trail_lock_trigger_pct=_parse_float(_read_env("LIVE_TRAIL_LOCK_TRIGGER_PCT", "0.08"), 0.08),
            trail_lock_pct=_parse_float(_read_env("LIVE_TRAIL_LOCK_PCT", "0.05"), 0.05),
            save_to_supabase=_parse_bool(_read_env("LIVE_SAVE_TO_SUPABASE", "0"), False),
            king_model_path=str(_read_env("LIVE_KING_MODEL_PATH", "api/models/KING_CRYPTO 👑.pkl")),
            council_model_path=str(_read_env("LIVE_COUNCIL_MODEL_PATH", "api/models/COUNCIL_CRYPTO.pkl")),
            max_open_positions=int(float(_read_env("LIVE_MAX_OPEN_POSITIONS", "5") or 5)),
        )

    def update_config(self, updates: Dict[str, Any]):
        with self._lock:
            if self._status == "running":
                raise RuntimeError("Cannot update config while bot is running. Stop it first.")
            
            # Apply updates
            current = asdict(self.config)
            
            # Handle list parsing for coins if string provided
            if "coins" in updates and isinstance(updates["coins"], str):
                updates["coins"] = _parse_coins(updates["coins"])
                
            for k, v in updates.items():
                if k in current:
                    # Type conversion
                    if k in ["king_threshold", "council_threshold", "max_notional_usd", "pct_cash_per_trade", 
                             "target_pct", "stop_loss_pct", "trail_be_pct", "trail_lock_trigger_pct", "trail_lock_pct"]:
                        current[k] = _parse_float(v, current[k])
                    elif k in ["bars_limit", "poll_seconds", "max_open_positions", "hold_max_bars"]:
                        current[k] = int(float(v))
                    elif k in ["use_council", "enable_sells", "use_trailing", "save_to_supabase"]:
                        current[k] = bool(v)
                    else:
                        current[k] = v
            
            self.config = BotConfig(**current)
            self._log("Configuration updated.")

    def start(self):
        with self._lock:
            if self._status == "running":
                self._log("Bot is already running.")
                return
            
            self._stop_event.clear()
            self._error_msg = None
            self._status = "starting"
            self._started_at = datetime.now(timezone.utc).isoformat()
            
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._status = "running"
            self._log("Bot started in background thread.")

    def stop(self):
        with self._lock:
            if self._status != "running":
                return
            self._log("Stopping bot...")
            self._status = "stopping"
            self._stop_event.set()
    
    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "config": asdict(self.config),
                "last_scan": self._last_scan_time,
                "error": self._error_msg,
                "data_stream": self._data_stream,
                "logs": list(self._logs)[-500:], # Return last 500 logs
                "trades": list(self._trades)[-50:],
                "started_at": self._started_at
            }

    def _load_models(self):
        from api.backtest_radar import load_model, reconstruct_meta_model
        from api.council_validator import load_council_validator_from_path

        self._log(f"Loading KING model from {self.config.king_model_path}...")
        king_art = load_model(self.config.king_model_path)
        if king_art is None:
            raise ValueError(f"Failed to load KING model from {self.config.king_model_path}")
        
        king_clf = reconstruct_meta_model(king_art)
        if king_clf is None:
             # Fallback if it's already a classifier
             king_clf = king_art

        self._log(f"Loading COUNCIL model from {self.config.council_model_path}...")
        # Note: load_council_validator_from_path handles its own loading internal to the pkl
        validator = load_council_validator_from_path(self.config.council_model_path)
        if validator is None:
            # We don't raise here because COUNCIL might be optional if use_council is False,
            # but usually it's better to have it if configured.
            self._log("Warning: COUNCIL validator failed to load.")

        return king_art, king_clf, validator


    def _include_open_positions_in_coins(self):
        """
        When starting, automatically include any currently open Alpaca positions in the scan list
        so the bot can manage exits (sell logic) for them.
        """
        try:
            positions = list(self.api.list_positions() or [])
        except Exception:
            return

        if not positions:
            return

        existing = list(self.config.coins or [])
        existing_norm = {_normalize_symbol(c) for c in existing}

        added = 0
        for p in positions:
            sym = _format_symbol_for_bot(str(getattr(p, "symbol", "") or ""))
            if not sym:
                continue
            n = _normalize_symbol(sym)
            if n in existing_norm:
                continue
            existing.append(sym)
            existing_norm.add(n)
            added += 1

        if added:
            self.config.coins = existing
            self._log(f"Included {added} open Alpaca position(s) into coin list.")

    def _sync_pos_state(self):
        """
        Sync internal _pos_state with actual Alpaca positions.
        Removes symbols from _pos_state that no longer have open positions in Alpaca.
        Ensures the total_managed count used by Risk Firewall is accurate.
        """
        try:
            positions = self.api.list_positions()
            current_alpaca_norms = set()
            
            for p in positions:
                sym = str(getattr(p, "symbol", "") or "")
                norm = _normalize_symbol(sym)
                current_alpaca_norms.add(norm)
                
                # If we have it in Alpaca but not in _pos_state, add a basic entry
                if norm not in self._pos_state:
                    self._pos_state[norm] = {
                        "entry_price": float(getattr(p, "avg_entry_price", 0)),
                        "entry_ts": datetime.now(timezone.utc).isoformat(),
                        "bars_held": 0,
                        "current_stop": None,
                        "trail_mode": "NONE",
                    }
            
            # Remove symbols from _pos_state that are NOT in Alpaca
            to_remove = [norm for norm in self._pos_state if norm not in current_alpaca_norms]
            for norm in to_remove:
                self._log(f"Sync: Removing {norm} from internal state (no longer in Alpaca).")
                self._pos_state.pop(norm, None)
                
        except Exception as e:
            self._log(f"Sync Error: Failed to synchronize position state: {e}")

    def _align_for_king(self, X_src: pd.DataFrame, king_artifact: object) -> pd.DataFrame:
        if not isinstance(X_src, pd.DataFrame):
            X_src = pd.DataFrame(X_src)

        if not isinstance(king_artifact, dict) or king_artifact.get("kind") != "meta_labeling_system":
            return X_src.replace([np.inf, -np.inf], np.nan).fillna(0)

        pm = king_artifact.get("primary_model") or {}
        feats = list(pm.get("feature_names") or [])
        if not feats:
            return X_src.replace([np.inf, -np.inf], np.nan).fillna(0)

        Xk = X_src.copy()
        missing = [c for c in feats if c not in Xk.columns]
        for c in missing:
            Xk[c] = 0

        Xk = Xk[feats]

        for col in Xk.columns:
            if not pd.api.types.is_numeric_dtype(Xk[col]):
                Xk[col] = pd.to_numeric(Xk[col], errors="coerce")

        Xk = Xk.replace([np.inf, -np.inf], np.nan).fillna(0)
        return Xk

    def _prepare_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        from api.train_exchange_model import add_technical_indicators, add_indicator_signals, add_massive_features

        df = bars.copy()
        # Original script logic
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            df = df.set_index("timestamp")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns])

        # Some TA feature generators crash on very short series (e.g., certain windowed indicators).
        # If we don't have enough bars, skip this scan safely.
        if len(df) < 50:
            self._log(f"Warning: Not enough bars for features ({len(df)} < 50)")
            return pd.DataFrame()

        try:
            feat = add_technical_indicators(df)
            if feat is None or feat.empty:
                return pd.DataFrame()
            feat = add_indicator_signals(feat)
            feat = add_massive_features(feat)
        except Exception as e:
            self._log(f"Feature generation error: {e}")
            return pd.DataFrame()

        feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        return feat

    def _process_bars(self, df: pd.DataFrame, symbol: str, source: str, limit: int) -> pd.DataFrame:
        """Standardize bar processing and update state for UI."""
        if df.empty:
            self._handle_fetch_error("Empty bars returned", symbol, limit, source)
            return df
            
        count = len(df)
        last_bar = df.iloc[-1]
        
        # Update data stream state for UI
        self._data_stream[symbol] = {
            "source": source,
            "count": count,
            "timestamp": last_bar["timestamp"].isoformat() if hasattr(last_bar["timestamp"], "isoformat") else str(last_bar["timestamp"]),
            "status": "OK" if count >= limit * 0.5 else "PARTIAL",
            "has_volume": float(last_bar.get("volume", 0)) > 0,
            "error": None
        }
        
        # Extra debug log for "all debugs" requirement
        self._log(f"DEBUG: {symbol} bars={count} src={source} last_close={last_bar['close']:.4f}")
        
        return df.tail(limit)

    def _handle_fetch_error(self, err: Any, symbol: str, limit: int, source: str = "unknown"):
        """Record and log data fetching errors."""
        err_msg = str(err)
        self._data_stream[symbol] = {
            "source": source,
            "count": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "ERROR",
            "has_volume": False,
            "error": err_msg
        }
        self._log(f"DATA ERROR: {symbol} via {source}: {err_msg}")

    def _get_bars(self, symbol: str, limit: int) -> pd.DataFrame:
        """Dispatcher to fetch bars based on symbol type and data source."""
        # Detect if it's likely a stock (no slash, 4 letters or explicitly dot-suffixed)
        is_crypto = "/" in symbol or symbol.endswith("USD") or symbol.endswith("USDT")
        
        # If the user explicitly sets a data source for the whole bot, we might respect it,
        # but for symbols like EGX ones, we need specific handling.
        source = (getattr(self.config, "data_source", "alpaca") or "alpaca").lower()
        
        # EGX Heuristic: 4 uppercase letters and no slash
        if not is_crypto and len(symbol) <= 5 and symbol.isupper() and "/" not in symbol:
            # Likely EGX or US Stock
            if source == "alpaca":
                 # Fallback to yfinance for EGX symbols as Alpaca doesn't support them
                 # and user report shows Alpaca Crypto API is being wrongly triggered.
                 self._log(f"Routing {symbol} to yfinance (Stock Detection)")
                 return self._get_yfinance_bars(symbol, limit)

        if symbol in self._data_stream and self._data_stream[symbol]["status"] == "ERROR" and "yfinance" not in self._data_stream[symbol].get("message", ""):
            # If already failed on primary, might already be routed or need routing
            pass

        if is_crypto:
            return self._get_crypto_bars(symbol, limit)
        elif source == "tvdata":
            return self._get_tvdata_bars(symbol, limit)
        else:
            return self._get_stock_bars(symbol, limit)

    def _get_crypto_bars(self, symbol: str, limit: int) -> pd.DataFrame:
        try:
            source = (getattr(self.config, "data_source", "alpaca") or "alpaca").lower()
            if source == "binance":
                from api.binance_data import fetch_binance_bars_df
                bars = fetch_binance_bars_df(symbol, timeframe=self.config.timeframe, limit=int(limit))
            else:
                # Alpaca Crypto V2
                bars = self.api.get_crypto_bars(symbol, timeframe=self.config.timeframe, limit=int(limit)).df
            
            return self._process_bars(bars, symbol, source, limit)
        except Exception as e:
            return self._handle_fetch_error(e, symbol, limit)

    def _get_stock_bars(self, symbol: str, limit: int) -> pd.DataFrame:
        """Fetch stock bars (US or mapped EGX)."""
        try:
            source = (getattr(self.config, "data_source", "alpaca") or "alpaca").lower()
            
            # Map timeframe for Alpaca Stocks
            tf_map = {"1Hour": "1Hour", "1Day": "1Day", "1min": "1Min", "5min": "5Min", "15min": "15Min"}
            tf = tf_map.get(self.config.timeframe, "1Day")

            if source == "alpaca":
                # Alpaca Stocks V2
                bars = self.api.get_bars(symbol, timeframe=tf, limit=int(limit)).df
                return self._process_bars(bars, symbol, source, limit)
            else:
                return self._get_yfinance_bars(symbol, limit)
        except Exception as e:
            return self._handle_fetch_error(e, symbol, limit)

    def _process_bars(self, bars: pd.DataFrame, symbol: str, source: str, limit: int) -> pd.DataFrame:
        if bars is None or bars.empty:
            self._data_stream[symbol] = {
                "source": source, "count": 0, "timestamp": datetime.now(timezone.utc).isoformat(), "status": "EMPTY"
            }
            return pd.DataFrame()

        has_volume = False
        if "volume" in bars.columns:
            try: has_volume = float(bars["volume"].sum()) > 0
            except: pass

        self._data_stream[symbol] = {
            "source": source,
            "count": len(bars),
            "has_volume": has_volume,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "OK"
        }
        return bars.reset_index().tail(limit)

    def _handle_fetch_error(self, e: Exception, symbol: str, limit: int) -> pd.DataFrame:
        err_msg = str(e)
        self._data_stream[symbol] = {
            "source": self.config.data_source,
            "count": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "ERROR",
            "message": err_msg
        }
        if "451" in str(e) or "block" in str(e).lower() or "invalid symbol" in str(e).lower():
             self._log(f"Fetch failed for {symbol}: {e}. Trying yfinance fallback...")
             return self._get_yfinance_bars(symbol, limit)
        self._log(f"Error fetching bars for {symbol}: {e}")
        return pd.DataFrame()

    def _get_yfinance_bars(self, symbol: str, limit: int) -> pd.DataFrame:
        try:
            # Fallback for EGX symbols in yfinance: add .CA if not present and no dot
            if symbol.isupper() and len(symbol) <= 5 and "." not in symbol and "/" not in symbol:
                yf_sym = f"{symbol}.CA"
            else:
                yf_sym = symbol.replace("/", "-")
            
            # Map timeframe
            tf_map = {"1Hour": "1h", "1min": "1m", "5min": "5m", "15min": "15m", "1Day": "1d", "1Day": "1d"}
            period_map = {"1h": "10d", "1m": "1d", "5m": "5d", "15m": "10d", "1d": "1mo"}
            
            tf = tf_map.get(self.config.timeframe, "1h")
            period = period_map.get(tf, "10d")
            
            ticker = yf.Ticker(yf_sym)
            df = ticker.history(period=period, interval=tf)
            if df.empty:
                return pd.DataFrame()
            
            df = df.reset_index()
            df = df.rename(columns={
                "Datetime": "timestamp", 
                "Date": "timestamp",
                "Open": "open", 
                "High": "high", 
                "Low": "low", 
                "Close": "close", 
                "Volume": "volume"
            })
            # Ensure UTC
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
                
            return self._process_bars(df, symbol, "yfinance", limit)
        except Exception as e:
            self._handle_fetch_error(e, symbol, limit)
            self._log(f"yfinance fallback failed for {symbol}: {e}")
            return pd.DataFrame()

    def _get_tvdata_bars(self, symbol: str, limit: int) -> pd.DataFrame:
        """Fetch bars using tvDatafeed (TradingView)."""
        if TvDatafeed is None:
            self._log("tvDatafeed not installed. Falling back to yfinance.")
            return self._get_yfinance_bars(symbol, limit)
        
        try:
            from api.tradingview_integration import get_tradingview_exchange
            tv_exchange = get_tradingview_exchange(symbol)
            base_sym = symbol.split(".")[0] if "." in symbol else symbol
            
            # Map timeframe
            tf_map = {
                "1min": Interval.in_1_minute,
                "5min": Interval.in_5_minute,
                "15min": Interval.in_15_minute,
                "30Min": Interval.in_30_minute,
                "1Hour": Interval.in_1_hour,
                "4Hour": Interval.in_4_hour,
                "1Day": Interval.in_daily
            }
            tf = tf_map.get(self.config.timeframe, Interval.in_1_hour)
            
            tv = TvDatafeed()
            df = tv.get_hist(symbol=base_sym, exchange=tv_exchange, interval=tf, n_bars=int(limit or 200))
            
            if df is None or df.empty:
                self._log(f"tvDatafeed returned no data for {symbol}. Trying yfinance.")
                return self._get_yfinance_bars(symbol, limit)
            
            df = df.reset_index().rename(columns={
                'datetime': 'timestamp', 
                'open': 'open', 
                'high': 'high', 
                'low': 'low', 
                'close': 'close', 
                'volume': 'volume'
            })
            
            # Ensure UTC
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
                
            return self._process_bars(df, symbol, "tvdata", limit)
        except Exception as e:
            self._log(f"tvDatafeed failed for {symbol}: {e}. Falling back to yfinance.")
            return self._get_yfinance_bars(symbol, limit)

    def _save_to_supabase(self, bars: pd.DataFrame, symbol: str):
        # Check if saving to Supabase is enabled
        if not getattr(self.config, "save_to_supabase", True):
            return
            
        if bars.empty:
            return

        try:
            # Crypto bars are intraday/time-series, so store them in `stock_bars_intraday`
            # (NOT `stock_prices`, which is daily and keyed by `date date`).
            exchange = "CRYPTO"
            timeframe = _to_intraday_timeframe(getattr(self.config, "timeframe", "") or "")
            
            rows = []
            # Make sure we have timestamp
            df = bars.reset_index() if "timestamp" not in bars.columns else bars.copy()
            if "timestamp" not in df.columns:
                 return

            for _, row in df.iterrows():
                ts = row["timestamp"]
                # Convert timestamp to ISO/String for DB timestamptz
                ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                
                record = {
                    "symbol": symbol.split("/")[0],
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "ts": ts_str,
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    # volume is bigint in schema; crypto volume can be fractional.
                    "volume": _safe_int(row.get("volume", 0), default=0),
                }
                rows.append(record)

            if rows:
                # Supabase/Postgres throws `21000 ... cannot affect row a second time` if a single
                # upsert statement contains duplicate values for the conflict key.
                # Alpaca bars can occasionally contain duplicate timestamps; dedupe by the conflict key.
                deduped: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
                for r in rows:
                    key = (str(r.get("symbol")), str(r.get("exchange")), str(r.get("timeframe")), str(r.get("ts")))
                    deduped[key] = r
                rows = list(deduped.values())

                _supabase_upsert_with_retry("stock_bars_intraday", rows, on_conflict="symbol,exchange,timeframe,ts")
                # self._log(f"Saved {len(rows)} bars for {symbol} to DB.")

        except Exception as e:
            self._log(f"DB Save Error {symbol}: {e}")

    def _has_open_position(self, symbol: str) -> bool:
        try:
            norm = _normalize_symbol(symbol)
            for p in self.api.list_positions():
                if _normalize_symbol(getattr(p, "symbol", "")) == norm:
                    return True
            return False
        except Exception:
            return False

    def _get_open_position(self, symbol: str):
        try:
            norm = _normalize_symbol(symbol)
            for p in self.api.list_positions():
                if _normalize_symbol(getattr(p, "symbol", "")) == norm:
                    return p
            return None
        except Exception:
            return None

    def _buy_market(self, symbol: str, notional_usd: float) -> bool:
        try:
            order = self.api.submit_order(
                symbol=symbol,
                notional=float(notional_usd),
                side="buy",
                type="market",
                time_in_force="gtc",
            )
            try:
                avg_fill = float(getattr(order, "filled_avg_price", 0) or 0) or None
            except Exception:
                avg_fill = None
            self._pos_state[_normalize_symbol(symbol)] = {
                "entry_price": avg_fill,
                "entry_ts": datetime.now(timezone.utc).isoformat(),
                "bars_held": 0,
                "current_stop": None,
                "trail_mode": "NONE",
            }
            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": "BUY",
                "amount": float(notional_usd),
                "price": avg_fill,
                "order_id": getattr(order, "id", "unknown")
            }
            self._trades.append(trade)
            self._save_trade_persistent(trade)
            
            if self.telegram_bridge:
                self.telegram_bridge.send_notification(
                    f"🟢 *BUY EXECUTED*\n\n"
                    f"Symbol: {symbol}\n"
                    f"Price: ${avg_fill:.2f}\n"
                    f"Amount: ${float(notional_usd):.2f}"
                )
            return True
        except Exception as e:
            self._log(f"Buy failed for {symbol}: {e}")
            return False

    def _sell_market(self, symbol: str, qty: float) -> bool:
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=float(qty),
                side="sell",
                type="market",
                time_in_force="gtc",
            )
            # Calculate PnL if possible
            pnl = 0.0
            entry_price = self._pos_state.get(_normalize_symbol(symbol), {}).get("entry_price")
            fill_price = None
            try:
                fill_price = float(getattr(order, "filled_avg_price", 0) or 0) or None
            except:
                pass
                
            if entry_price and fill_price:
                pnl = (fill_price - entry_price) * float(qty)

            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": "SELL",
                "amount": float(qty),
                "price": fill_price,
                "entry_price": entry_price,
                "pnl": pnl,
                "order_id": getattr(order, "id", "unknown"),
            }
            self._trades.append(trade)
            self._save_trade_persistent(trade)
            
            if self.telegram_bridge:
                emoji = "🟢" if pnl > 0 else "🔴"
                self.telegram_bridge.send_notification(
                    f"{emoji} *SELL EXECUTED*\n\n"
                    f"Symbol: {symbol}\n"
                    f"Exit Price: ${fill_price:.2f}\n"
                    f"PnL: ${pnl:.2f}"
                )
            
            self._pos_state.pop(_normalize_symbol(symbol), None)
            return True
        except Exception as e:
            self._log(f"Sell failed for {symbol}: {e}")
            return False

    def _maybe_sell_position(self, symbol: str, bars: pd.DataFrame) -> bool:
        """
        Apply backtest-style exits on the latest bar:
        - Stop loss
        - Target
        - Trailing stop updates (BE / lock)
        - Time exit by bars held
        Returns True if a sell was sent.
        """
        if not self.config.enable_sells:
            return False

        pos = self._get_open_position(symbol)
        if pos is None:
            self._pos_state.pop(_normalize_symbol(symbol), None)
            return False

        if bars is None or bars.empty:
            return False

        try:
            last = bars.iloc[-1]
            hi = float(last.get("high", 0) or 0)
            lo = float(last.get("low", 0) or 0)
            close = float(last.get("close", 0) or 0)
        except Exception:
            return False

        state = self._pos_state.get(_normalize_symbol(symbol)) or {}
        entry_price = state.get("entry_price")
        if not entry_price:
            try:
                entry_price = float(getattr(pos, "avg_entry_price", 0) or 0) or None
            except Exception:
                entry_price = None

        if not entry_price or entry_price <= 0:
            return False

        take_profit = float(entry_price) * (1 + float(self.config.target_pct))
        stop_loss = float(entry_price) * (1 - float(self.config.stop_loss_pct))
        current_stop = state.get("current_stop")
        if current_stop is None:
            current_stop = float(stop_loss)
        trail_mode = state.get("trail_mode") or "NONE"

        # Conservative bar evaluation using current stop/target
        if lo <= float(current_stop):
            qty = float(getattr(pos, "qty", 0) or 0) if pos else (state.get("notional", 100) / entry_price)
            self._log(f"{symbol}: SELL (STOP) qty={qty} stop={current_stop:.6f} entry={entry_price:.6f}")
            if self.config.execution_mode == "TELEGRAM":
                self._save_signal_record(symbol, current_stop, qty * current_stop, 0, None, action="SELL")
                self._pos_state.pop(_normalize_symbol(symbol), None)
                self._log(f"SKIPPING Alpaca Sell (Telegram-Only Mode)")
                return True # Count as handled
            return self._sell_market(symbol, qty=qty)

        if hi >= float(take_profit):
            qty = float(getattr(pos, "qty", 0) or 0) if pos else (state.get("notional", 100) / entry_price)
            self._log(f"{symbol}: SELL (TARGET) qty={qty} tp={take_profit:.6f} entry={entry_price:.6f}")
            if self.config.execution_mode == "TELEGRAM":
                self._save_signal_record(symbol, take_profit, qty * take_profit, 0, None, action="SELL")
                self._pos_state.pop(_normalize_symbol(symbol), None)
                self._log(f"SKIPPING Alpaca Sell (Telegram-Only Mode)")
                return True # Count as handled
            return self._sell_market(symbol, qty=qty)

        # Trailing stop updates (effective next bar)
        if self.config.use_trailing:
            be_price = float(entry_price)
            lock_price = float(entry_price) * (1 + float(self.config.trail_lock_pct))
            if hi >= float(entry_price) * (1 + float(self.config.trail_lock_trigger_pct)) and float(current_stop) < lock_price:
                current_stop = lock_price
                trail_mode = "+LOCK"
            elif hi >= float(entry_price) * (1 + float(self.config.trail_be_pct)) and float(current_stop) < be_price:
                current_stop = be_price
                trail_mode = "BE"

        bars_held = int(state.get("bars_held") or 0) + 1
        if bars_held >= int(self.config.hold_max_bars):
            qty = float(getattr(pos, "qty", 0) or 0) if pos else (state.get("notional", 100) / entry_price)
            self._log(f"{symbol}: SELL (TIME) qty={qty} bars={bars_held} close={close:.6f}")
            if self.config.execution_mode == "TELEGRAM":
                self._save_signal_record(symbol, close, qty * close, 0, None, action="SELL")
                self._pos_state.pop(_normalize_symbol(symbol), None)
                self._log(f"SKIPPING Alpaca Sell (Telegram-Only Mode)")
                return True
            return self._sell_market(symbol, qty=qty)

        self._pos_state[_normalize_symbol(symbol)] = {
            **state,
            "entry_price": entry_price,
            "bars_held": bars_held,
            "current_stop": float(current_stop),
            "trail_mode": trail_mode,
        }
        return False

    def _run_loop(self):
        try:
            self._log("Initializing models and connection...")

            if not self.config.alpaca_key_id or not self.config.alpaca_secret_key:
                 raise ValueError("Alpaca API keys missing in config/env")

            self.api = create_alpaca_client(
                key_id=self.config.alpaca_key_id,
                secret_key=self.config.alpaca_secret_key,
                base_url=self.config.alpaca_base_url,
                logger=self._log,
            )

            # Auto-include open positions so sells are managed even if not in LIVE_COINS.
            self._include_open_positions_in_coins()

            self.king_obj, self.king_clf, self.validator = self._load_models()
            self._log(f"Models loaded. Polling every {self.config.poll_seconds}s.")
            self._log(f"Coins: {', '.join(self.config.coins)}")
            self._log(f"Thresholds: KING>={self.config.king_threshold}, COUNCIL>={self.config.council_threshold}")

            while not self._stop_event.is_set():
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
                self._last_scan_time = now
                self._log(f"--- SCAN CYCLE START ({now}) ---")
                
                # Sync positions first to ensure Risk Firewall is accurate
                self._sync_pos_state()
                
                self._log(f"Config: {self.config.timeframe} | {len(self.config.coins)} symbols | mode={self.config.execution_mode} | active_positions={len(self._pos_state)}/{self.config.max_open_positions}")
                
                for symbol in self.config.coins:
                    if self._stop_event.is_set():
                        break
                        
                    bars = self._get_bars(symbol, limit=self.config.bars_limit)
                    if bars.empty:
                        self._log(f"{symbol}: No bars found.")
                        continue

                    # Save to Supabase (Background or Sync? Sync is safer for data integrity, threaded is faster)
                    # Doing sync for now as poll_seconds is usually high (300s)
                    self._save_to_supabase(bars, symbol)

                    # If a position exists, apply sell logic first using the latest bar, then skip buy logic.
                    if self._has_open_position(symbol):
                        sold = self._maybe_sell_position(symbol, bars)
                        if sold:
                            continue
                        continue

                    features = self._prepare_features(bars)
                    if features.empty:
                        self._log(f"{symbol}: Features empty (insufficient data).")
                        continue

                    X_all = features.iloc[[-1]].copy()
                    Xk = self._align_for_king(X_all, self.king_obj)

                    try:
                        king_conf = float(self.king_clf.predict_proba(Xk)[:, 1][0])
                    except Exception:
                        continue

                    if king_conf < self.config.king_threshold:
                        self._log(f"{symbol}: KING pass ({king_conf:.2f} < {self.config.king_threshold})")
                        continue

                    self._log(f"SIGNAL: {symbol} KING={king_conf:.2f}")

                    if self.config.use_council:
                        try:
                            council_conf = float(self.validator.predict_proba(X_all, primary_conf=np.asarray([king_conf]))[:, 1][0])
                        except Exception:
                            continue

                        self._log(f"COUNCIL CHECK: {symbol} COUNCIL={council_conf:.2f}")

                        if council_conf < self.config.council_threshold:
                            self._log(f"COUNCIL REJECTED: {symbol} ({council_conf:.2f} < {self.config.council_threshold})")
                            continue
                    else:
                         self._log(f"COUNCIL SKIPPED: {symbol}")
                         council_conf = None

                    # --- RISK FIREWALL: Max Open Trades ---
                    try:
                        # Count total positions (real + virtual tracked in _pos_state)
                        total_managed = len(self._pos_state)
                        if total_managed >= self.config.max_open_positions:
                            msg = f"⚠️ *RISK FIREWALL ALERT*\n\nLimit Reached: {total_managed}/{self.config.max_open_positions} positions.\nSkipping signal for `{symbol}`.\n\n_Increase 'Max Open Trades' in settings if you want to allow more simultaneous positions._"
                            self._log(f"RISK FIREWALL: Max positions reached ({total_managed}/{self.config.max_open_positions}). Skipping signal for {symbol}.")
                            if self.telegram_bridge:
                                self.telegram_bridge.send_notification(msg)
                            continue
                    except Exception as e:
                        self._log(f"Error checking Risk Firewall: {e}")

                    # Signal Confirmed -> Send better Telegram notification
                    last_price = float(bars.iloc[-1]['close'])
                    
                    try:
                        account = self.api.get_account()
                        cash = float(getattr(account, "cash", 0) or 0)
                    except Exception:
                        cash = 0.0
                    
                    notional = min(cash * self.config.pct_cash_per_trade, self.config.max_notional_usd)
                    
                    # Minimum notional check
                    if notional < 10:
                        self._log(f"{symbol}: insufficient cash ({cash:.2f})")
                        continue

                    self._send_telegram_signal(symbol, last_price, notional, king_conf, council_conf)
                    
                    # Store signal record
                    self._save_signal_record(symbol, last_price, notional, king_conf, council_conf, action="BUY")

                    # Record position state for exit management (real or virtual)
                    self._pos_state[_normalize_symbol(symbol)] = {
                        "entry_price": last_price,
                        "entry_ts": datetime.now(timezone.utc).isoformat(),
                        "bars_held": 0,
                        "current_stop": None,
                        "trail_mode": "NONE",
                        "notional": notional # Track for sell sizing
                    }

                    # Execute real trade if not in Telegram-only mode
                    if self.config.execution_mode != "TELEGRAM":
                        self._log(f"{symbol}: Placing BUY order for ${notional:.2f}")
                        try:
                            ok = self._buy_market(symbol, notional_usd=notional)
                            if ok:
                                self._log(f"{symbol}: BUY executed.")
                            else:
                                self._log(f"{symbol}: BUY failed.")
                        except Exception as ex:
                            self._log(f"Error executing buy for {symbol}: {ex}")
                    else:
                        self._log(f"{symbol}: Telegram-Only Signal recorded.")

                # Wait for next poll or stop signal
                # Break it into small chunks to be responsive to stop
                for _ in range(self.config.poll_seconds):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
            
            self._log("Bot loop ended.")
            self._status = "stopped"

        except Exception as e:
            err = traceback.format_exc()
            self._log(f"CRITICAL ERROR: {e}")
            self._log(err)
            self._error_msg = str(e)
            self._status = "error"

# Bot Manager to handle multiple bots
class BotManager:
    def __init__(self):
        self._bots: Dict[str, LiveBot] = {}
        self._state_file = Path("state/bots.json")
        self._telegram_bridge = None
        self._load_bots()

    def set_telegram_bridge(self, bridge):
        self._telegram_bridge = bridge
        for bot in self._bots.values():
            bot.set_telegram_bridge(bridge)

    def _load_bots(self):
        """Load bot configurations from state/bots.json."""
        if self._state_file.exists():
            try:
                with open(self._state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for bot_id, config_dict in data.items():
                        # Handle list to dict migration if needed
                        if "coins" in config_dict and isinstance(config_dict["coins"], str):
                            config_dict["coins"] = _parse_coins(config_dict["coins"])
                        
                        cfg = BotConfig(**config_dict)
                        self._bots[bot_id] = LiveBot(bot_id=bot_id, config=cfg)
            except Exception as e:
                print(f"Error loading bots: {e}")
        
        # Ensure a 'primary' bot exists if none loaded
        if "primary" not in self._bots:
            self.create_bot("primary", "Primary Bot")

    def save_bots(self):
        """Save all bot configurations."""
        try:
            self._state_file.parent.mkdir(exist_ok=True)
            data = {bid: asdict(bot.config) for bid, bot in self._bots.items()}
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving bots: {e}")

    def create_bot(self, bot_id: str, name: str) -> LiveBot:
        if bot_id in self._bots:
            raise ValueError(f"Bot ID {bot_id} already exists.")
        
        # Start with default config
        bot = LiveBot(bot_id=bot_id)
        bot.config.name = name
        if self._telegram_bridge:
            bot.set_telegram_bridge(self._telegram_bridge)
            
        self._bots[bot_id] = bot
        self.save_bots()
        return bot

    def delete_bot(self, bot_id: str):
        if bot_id == "primary":
            raise ValueError("Cannot delete primary bot.")
        if bot_id in self._bots:
            self._bots[bot_id].stop()
            del self._bots[bot_id]
            self.save_bots()

    def get_bot(self, bot_id: str) -> Optional[LiveBot]:
        return self._bots.get(bot_id)

    def list_bots(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": bid,
                "name": bot.config.name,
                "status": bot.get_status()["status"],
                "mode": bot.config.execution_mode
            }
            for bid, bot in self._bots.items()
        ]

# Global Manager Instance
bot_manager = BotManager()

# Backward compatibility for single-bot logic
bot_instance = bot_manager.get_bot("primary")
