import os
import time
import threading
import warnings
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from collections import deque

import numpy as np
import pandas as pd

# We'll use the existing imports from run_live_bot.py logic
# assuming api module structure is available.
from api.stock_ai import _supabase_upsert_with_retry
from api.alpaca_adapter import create_alpaca_client

warnings.filterwarnings("ignore")

@dataclass
class BotConfig:
    alpaca_key_id: str
    alpaca_secret_key: str
    alpaca_base_url: str
    coins: list[str]
    king_threshold: float
    council_threshold: float
    max_notional_usd: float
    pct_cash_per_trade: float
    bars_limit: int
    poll_seconds: int
    timeframe: str = "1Hour"
    use_council: bool = True
    data_source: str = "binance"  # "alpaca" | "binance" | "yfinance"

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
    def __init__(self):
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
        self.config = self._build_default_config()
        
        # Models
        self.king_obj = None
        self.king_clf = None
        self.validator = None
        self.api = None
        # Per-symbol runtime state for exit logic
        self._pos_state: Dict[str, Dict[str, Any]] = {}

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        with self._lock:
            print(line) # Keep printing to stdout for convenience
            self._logs.append(line)

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
                "logs": list(self._logs)[-50:], # Return last 50 logs
                "trades": list(self._trades)[-20:],
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

    def _get_crypto_bars(self, symbol: str, limit: int) -> pd.DataFrame:
        try:
            if (getattr(self.config, "data_source", "alpaca") or "alpaca").lower() == "binance":
                from api.binance_data import fetch_binance_bars_df

                bars = fetch_binance_bars_df(symbol, timeframe=self.config.timeframe, limit=int(limit))
            else:
                # Alpaca (self.api is set in _run_loop)
                bars = self.api.get_crypto_bars(symbol, timeframe=self.config.timeframe, limit=int(limit)).df
            
            if bars is None: return pd.DataFrame() # fail-safe
            
            # Update data stream
            self._data_stream[symbol] = {
                "source": self.config.data_source,
                "count": len(bars),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "OK" if not bars.empty else "EMPTY"
            }
            
            if bars.empty:
                return pd.DataFrame()
            return bars.reset_index()
        except Exception as e:
            err_msg = str(e)
            self._data_stream[symbol] = {
                "source": self.config.data_source,
                "count": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "ERROR",
                "message": err_msg
            }
            if "451" in str(e) or "block" in str(e).lower():
                 self._log(f"Binance blocked (451). Trying yfinance fallback...")
                 return self._get_yfinance_bars(symbol, limit)
            self._log(f"Error fetching bars for {symbol}: {e}")
            return pd.DataFrame()

    def _get_yfinance_bars(self, symbol: str, limit: int) -> pd.DataFrame:
        try:
            import yfinance as yf
            # Map BTC/USD to BTC-USD
            yf_sym = symbol.replace("/", "-")
            
            # Map timeframe
            tf_map = {"1Hour": "1h", "1min": "1m", "5min": "5m", "15min": "15m", "1Day": "1d"}
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
                
            return df.tail(limit)
        except Exception as e:
            self._log(f"yfinance fallback failed for {symbol}: {e}")
            return pd.DataFrame()

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
            self._trades.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": "BUY",
                "amount": notional_usd,
                "order_id": getattr(order, "id", "unknown")
            })
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
            self._trades.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": "SELL",
                "amount": float(qty),
                "order_id": getattr(order, "id", "unknown"),
            })
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
            qty = float(getattr(pos, "qty", 0) or 0)
            if qty > 0:
                self._log(f"{symbol}: SELL (STOP) qty={qty} stop={current_stop:.6f} entry={entry_price:.6f}")
                return self._sell_market(symbol, qty=qty)
            return False

        if hi >= float(take_profit):
            qty = float(getattr(pos, "qty", 0) or 0)
            if qty > 0:
                self._log(f"{symbol}: SELL (TARGET) qty={qty} tp={take_profit:.6f} entry={entry_price:.6f}")
                return self._sell_market(symbol, qty=qty)
            return False

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
            qty = float(getattr(pos, "qty", 0) or 0)
            if qty > 0:
                self._log(f"{symbol}: SELL (TIME) qty={qty} bars={bars_held} close={close:.6f}")
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
                self._log(f"Scanning... {now}")
                
                for symbol in self.config.coins:
                    if self._stop_event.is_set():
                        break
                        
                    bars = self._get_crypto_bars(symbol, limit=self.config.bars_limit)
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
                            continue
                    else:
                         self._log(f"COUNCIL SKIPPED: {symbol}")

                    try:
                        account = self.api.get_account()
                        cash = float(getattr(account, "cash", 0) or 0)
                    except Exception:
                        cash = 0.0

                    # Check Max Open Positions limit
                    try:
                        positions = self.api.list_positions()
                        if len(positions) >= self.config.max_open_positions:
                            # self._log(f"Scan: Max open positions reached ({len(positions)}). Skipping buys.")
                            pass
                        else:
                            notional = min(cash * self.config.pct_cash_per_trade, self.config.max_notional_usd)
                            if notional < 10:
                                self._log(f"{symbol}: insufficient cash ({cash:.2f})")
                                continue

                            self._log(f"{symbol}: Placing BUY order for ${notional:.2f}")
                            ok = self._buy_market(symbol, notional_usd=notional)
                            if ok:
                                self._log(f"{symbol}: BUY executed.")
                            else:
                                self._log(f"{symbol}: BUY failed.")
                    except Exception as e:
                        self._log(f"Error checking positions/buying for {symbol}: {e}")

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

# Global instance
bot_instance = LiveBot()
