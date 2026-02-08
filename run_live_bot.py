import os
import sys
import time
import warnings
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
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
    timeframe: str

    # Exit / risk management (optional, enabled by default)
    enable_sells: bool
    target_pct: float
    stop_loss_pct: float
    hold_max_bars: int
    use_trailing: bool
    trail_be_pct: float
    trail_lock_trigger_pct: float
    trail_lock_pct: float

    # Safety / behavior
    state_path: str


def _read_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _parse_float(value: str, default: float) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


def _parse_coins(raw: str) -> list[str]:
    coins = []
    for part in (raw or "").split(","):
        s = part.strip().upper()
        if not s:
            continue
        coins.append(s)
    return coins


def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper().replace("/", "")


def _format_symbol_for_bot(symbol: str) -> str:
    """
    Alpaca positions may return crypto like 'BTCUSD'. The bot/data calls generally use 'BTC/USD'.
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


def _build_config() -> BotConfig:
    coins = _parse_coins(
        _read_env(
            "LIVE_COINS",
            "BTC/USD,ETH/USD,LTC/USD,DOGE/USD",
        )
    )

    key_id = _read_env("ALPACA_KEY_ID", None) or _read_env("ALPACA_API_KEY", None)
    secret = _read_env("ALPACA_SECRET_KEY", None)
    if not key_id:
        raise RuntimeError("Missing required env var: ALPACA_KEY_ID (or ALPACA_API_KEY)")
    if not secret:
        raise RuntimeError("Missing required env var: ALPACA_SECRET_KEY")

    return BotConfig(
        alpaca_key_id=str(key_id),
        alpaca_secret_key=str(secret),
        alpaca_base_url=str(_read_env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")),
        coins=coins,
        king_threshold=_parse_float(_read_env("KING_THRESHOLD", "0.60"), 0.60),
        council_threshold=_parse_float(_read_env("COUNCIL_THRESHOLD", "0.35"), 0.35),
        max_notional_usd=_parse_float(_read_env("MAX_NOTIONAL_USD", "500"), 500.0),
        pct_cash_per_trade=_parse_float(_read_env("PCT_CASH_PER_TRADE", "0.10"), 0.10),
        bars_limit=int(float(_read_env("BARS_LIMIT", "200") or 200)),
        poll_seconds=int(float(_read_env("POLL_SECONDS", "300") or 300)),
        timeframe=str(_read_env("TIMEFRAME", "1Hour") or "1Hour"),
        enable_sells=_parse_bool(_read_env("LIVE_ENABLE_SELLS", "1"), True),
        target_pct=_parse_float(_read_env("LIVE_TARGET_PCT", "0.10") or "0.10", 0.10),
        stop_loss_pct=_parse_float(_read_env("LIVE_STOP_LOSS_PCT", "0.05") or "0.05", 0.05),
        hold_max_bars=int(float(_read_env("LIVE_HOLD_MAX_BARS", "20") or 20)),
        use_trailing=_parse_bool(_read_env("LIVE_USE_TRAILING", "1"), True),
        trail_be_pct=_parse_float(_read_env("LIVE_TRAIL_BE_PCT", "0.05") or "0.05", 0.05),
        trail_lock_trigger_pct=_parse_float(_read_env("LIVE_TRAIL_LOCK_TRIGGER_PCT", "0.08") or "0.08", 0.08),
        trail_lock_pct=_parse_float(_read_env("LIVE_TRAIL_LOCK_PCT", "0.05") or "0.05", 0.05),
        state_path=str(_read_env("LIVE_STATE_PATH", "run_live_bot_state.json") or "run_live_bot_state.json"),
    )


def _load_models():
    from api.backtest_radar import load_model, reconstruct_meta_model
    from api.council_validator import load_council_validator_from_path

    king_path = os.path.join("api", "models", "KING_CRYPTO 👑.pkl")
    council_path = os.path.join("api", "models", "COUNCIL_CRYPTO.pkl")

    king_obj = load_model(king_path)
    if king_obj is None:
        raise RuntimeError(f"Failed to load KING model at {king_path}")

    king_clf = king_obj
    if isinstance(king_obj, dict) and king_obj.get("kind") == "meta_labeling_system":
        king_clf = reconstruct_meta_model(king_obj)
    if king_clf is None or not hasattr(king_clf, "predict_proba"):
        raise RuntimeError("KING model does not support predict_proba()")

    validator = load_council_validator_from_path(council_path)
    if validator is None:
        raise RuntimeError(f"Failed to load Council Validator at {council_path}")

    return king_obj, king_clf, validator


def _align_for_king(X_src: pd.DataFrame, king_artifact: object) -> pd.DataFrame:
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


def _prepare_features(bars: pd.DataFrame) -> pd.DataFrame:
    from api.train_exchange_model import add_technical_indicators, add_indicator_signals, add_massive_features

    df = bars.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns])

    feat = add_technical_indicators(df)
    if feat is None or feat.empty:
        return pd.DataFrame()
    feat = add_indicator_signals(feat)
    feat = add_massive_features(feat)

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
    return feat


def _get_crypto_bars(api, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    try:
        # alpaca_trade_api returns a multi-index DataFrame; reset_index gives columns:
        # symbol, timestamp, open, high, low, close, volume, trade_count, vwap
        bars = api.get_crypto_bars(symbol, timeframe=str(timeframe or "1Hour"), limit=int(limit)).df
        if bars is None or getattr(bars, "empty", True):
            return pd.DataFrame()
        out = bars.reset_index()
        return out
    except Exception:
        return pd.DataFrame()


def _has_open_position(api, symbol: str) -> bool:
    try:
        norm = _normalize_symbol(symbol)
        for p in api.list_positions():
            if _normalize_symbol(getattr(p, "symbol", "")) == norm:
                return True
        return False
    except Exception:
        return False


def _get_open_position(api, symbol: str):
    try:
        norm = _normalize_symbol(symbol)
        for p in api.list_positions():
            if _normalize_symbol(getattr(p, "symbol", "")) == norm:
                return p
        return None
    except Exception:
        return None


def _buy_market(api, symbol: str, notional_usd: float):
    try:
        # notional is supported for crypto on Alpaca; if not, this will raise and we fall back to qty.
        order = api.submit_order(
            symbol=symbol,
            notional=float(notional_usd),
            side="buy",
            type="market",
            time_in_force="gtc",
        )
        return order
    except Exception:
        return None


def _sell_market(api, symbol: str, qty: float):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=float(qty),
            side="sell",
            type="market",
            time_in_force="gtc",
        )
        return order
    except Exception:
        return None


def _load_state(path: str) -> dict:
    try:
        if not path:
            return {}
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        return {}
    except Exception:
        return {}


def _save_state(path: str, state: dict) -> None:
    try:
        if not path:
            return
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        return


def _include_open_positions_in_coins(api, coins: list[str]) -> list[str]:
    """
    Ensure any existing open positions are managed by the bot even if not listed in LIVE_COINS.
    """
    try:
        have = {_normalize_symbol(c) for c in (coins or [])}
        out = list(coins or [])
        for p in api.list_positions():
            sym = _format_symbol_for_bot(getattr(p, "symbol", "") or "")
            if not sym:
                continue
            n = _normalize_symbol(sym)
            if n not in have:
                out.append(sym)
                have.add(n)
        return out
    except Exception:
        return list(coins or [])


def _maybe_sell_position(*, api, cfg: BotConfig, symbol: str, pos, bars: pd.DataFrame, state: dict) -> bool:
    """
    Backtest-style exits on the latest bar:
    - Stop loss
    - Target
    - Trailing stop updates (BE / lock)
    - Time exit by bars held
    Returns True if a sell was sent.
    """
    if not cfg.enable_sells:
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

    key = _normalize_symbol(symbol)
    sym_state = state.get(key) if isinstance(state, dict) else None
    if not isinstance(sym_state, dict):
        sym_state = {}

    entry_price = sym_state.get("entry_price")
    if not entry_price:
        try:
            entry_price = float(getattr(pos, "avg_entry_price", 0) or 0) or None
        except Exception:
            entry_price = None

    if not entry_price or float(entry_price) <= 0:
        return False

    take_profit = float(entry_price) * (1 + float(cfg.target_pct))
    stop_loss = float(entry_price) * (1 - float(cfg.stop_loss_pct))
    current_stop = sym_state.get("current_stop")
    if current_stop is None:
        current_stop = float(stop_loss)
    trail_mode = sym_state.get("trail_mode") or "NONE"

    # Conservative bar evaluation using current stop/target
    if lo <= float(current_stop):
        try:
            qty = float(getattr(pos, "qty", 0) or 0)
        except Exception:
            qty = 0.0
        if qty > 0:
            order = _sell_market(api, symbol, qty=qty)
            if order is not None:
                print(f"{symbol}: SELL (STOP) qty={qty} stop={float(current_stop):.6f} entry={float(entry_price):.6f}")
                state.pop(key, None)
                return True
        return False

    if hi >= float(take_profit):
        try:
            qty = float(getattr(pos, "qty", 0) or 0)
        except Exception:
            qty = 0.0
        if qty > 0:
            order = _sell_market(api, symbol, qty=qty)
            if order is not None:
                print(f"{symbol}: SELL (TARGET) qty={qty} tp={float(take_profit):.6f} entry={float(entry_price):.6f}")
                state.pop(key, None)
                return True
        return False

    # Trailing stop updates (effective next bar)
    if cfg.use_trailing:
        be_price = float(entry_price)
        lock_price = float(entry_price) * (1 + float(cfg.trail_lock_pct))
        if hi >= float(entry_price) * (1 + float(cfg.trail_lock_trigger_pct)) and float(current_stop) < lock_price:
            current_stop = lock_price
            trail_mode = "+LOCK"
        elif hi >= float(entry_price) * (1 + float(cfg.trail_be_pct)) and float(current_stop) < be_price:
            current_stop = be_price
            trail_mode = "BE"

    bars_held = int(sym_state.get("bars_held") or 0) + 1
    if bars_held >= int(cfg.hold_max_bars):
        try:
            qty = float(getattr(pos, "qty", 0) or 0)
        except Exception:
            qty = 0.0
        if qty > 0:
            order = _sell_market(api, symbol, qty=qty)
            if order is not None:
                print(f"{symbol}: SELL (TIME) qty={qty} bars={bars_held} close={close:.6f}")
                state.pop(key, None)
                return True

    state[key] = {
        **sym_state,
        "entry_price": float(entry_price),
        "bars_held": int(bars_held),
        "current_stop": float(current_stop),
        "trail_mode": str(trail_mode),
    }
    return False


def main() -> int:
    cfg = _build_config()
    state = _load_state(cfg.state_path)

    print("Initializing live bot (paper trading).")
    print(f"Coins: {', '.join(cfg.coins)}")
    print(f"Thresholds: KING >= {cfg.king_threshold:.2f}, COUNCIL >= {cfg.council_threshold:.2f}")
    print(f"Polling every {cfg.poll_seconds}s, bars_limit={cfg.bars_limit}, timeframe={cfg.timeframe}")
    if cfg.enable_sells:
        print(
            "Sells enabled: "
            f"TP={cfg.target_pct:.2%} SL={cfg.stop_loss_pct:.2%} "
            f"hold_max_bars={cfg.hold_max_bars} trailing={cfg.use_trailing}"
        )

    try:
        from api.alpaca_adapter import create_alpaca_client
    except Exception as e:
        print(f"Failed to import Alpaca client adapter: {e}")
        return 2

    try:
        king_obj, king_clf, validator = _load_models()
    except Exception as e:
        print(f"Model load failed: {e}")
        return 2

    api = create_alpaca_client(
        key_id=cfg.alpaca_key_id,
        secret_key=cfg.alpaca_secret_key,
        base_url=cfg.alpaca_base_url,
        logger=print,
    )

    while True:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        print(f"\nScan @ {now}")

        coins = _include_open_positions_in_coins(api, cfg.coins)
        for symbol in coins:
            bars = _get_crypto_bars(api, symbol, timeframe=cfg.timeframe, limit=cfg.bars_limit)
            if bars.empty:
                continue

            pos = _get_open_position(api, symbol)
            if pos is not None:
                sold = _maybe_sell_position(api=api, cfg=cfg, symbol=symbol, pos=pos, bars=bars, state=state)
                _save_state(cfg.state_path, state)
                if sold:
                    continue
                print(f"{symbol}: position open, holding")
                continue

            features = _prepare_features(bars)
            if features.empty:
                continue

            X_all = features.iloc[[-1]].copy()
            Xk = _align_for_king(X_all, king_obj)

            try:
                king_conf = float(king_clf.predict_proba(Xk)[:, 1][0])
            except Exception:
                continue

            if king_conf < cfg.king_threshold:
                print(f"{symbol}: KING fail ({king_conf:.2f})")
                continue

            try:
                council_conf = float(validator.predict_proba(X_all, primary_conf=np.asarray([king_conf]))[:, 1][0])
            except Exception:
                continue

            print(f"{symbol}: KING={king_conf:.2f} COUNCIL={council_conf:.2f}")

            if council_conf < cfg.council_threshold:
                continue

            try:
                account = api.get_account()
                cash = float(getattr(account, "cash", 0) or 0)
            except Exception:
                cash = 0.0

            notional = min(cash * cfg.pct_cash_per_trade, cfg.max_notional_usd)
            if notional < 10:
                print(f"{symbol}: insufficient cash ({cash:.2f})")
                continue

            order = _buy_market(api, symbol, notional_usd=notional)
            if order is not None:
                try:
                    avg_fill = float(getattr(order, "filled_avg_price", 0) or 0) or None
                except Exception:
                    avg_fill = None
                state[_normalize_symbol(symbol)] = {
                    "entry_price": avg_fill,
                    "entry_ts": datetime.now(timezone.utc).isoformat(),
                    "bars_held": 0,
                    "current_stop": None,
                    "trail_mode": "NONE",
                }
                _save_state(cfg.state_path, state)
                print(f"{symbol}: BUY (notional ${notional:.2f})")
            else:
                print(f"{symbol}: BUY failed")

        print(f"Sleeping {cfg.poll_seconds}s...")
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
