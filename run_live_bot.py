import os
import sys
import time
import warnings
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


def _read_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    v = os.getenv(name, default)
    if required and not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


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
        king_threshold=_parse_float(_read_env("KING_THRESHOLD", "0.06"), 0.60),
        council_threshold=_parse_float(_read_env("COUNCIL_THRESHOLD", "0.35"), 0.35),
        max_notional_usd=_parse_float(_read_env("MAX_NOTIONAL_USD", "500"), 500.0),
        pct_cash_per_trade=_parse_float(_read_env("PCT_CASH_PER_TRADE", "0.10"), 0.10),
        bars_limit=int(float(_read_env("BARS_LIMIT", "200") or 200)),
        poll_seconds=int(float(_read_env("POLL_SECONDS", "300") or 300)),
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


def _get_crypto_bars(api, symbol: str, limit: int) -> pd.DataFrame:
    try:
        # alpaca_trade_api returns a multi-index DataFrame; reset_index gives columns:
        # symbol, timestamp, open, high, low, close, volume, trade_count, vwap
        bars = api.get_crypto_bars(symbol, timeframe="1Hour", limit=int(limit)).df
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


def _buy_market(api, symbol: str, notional_usd: float) -> bool:
    try:
        # notional is supported for crypto on Alpaca; if not, this will raise and we fall back to qty.
        api.submit_order(
            symbol=symbol,
            notional=float(notional_usd),
            side="buy",
            type="market",
            time_in_force="gtc",
        )
        return True
    except Exception:
        return False


def main() -> int:
    cfg = _build_config()

    print("Initializing live bot (paper trading).")
    print(f"Coins: {', '.join(cfg.coins)}")
    print(f"Thresholds: KING >= {cfg.king_threshold:.2f}, COUNCIL >= {cfg.council_threshold:.2f}")
    print(f"Polling every {cfg.poll_seconds}s, bars_limit={cfg.bars_limit}")

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

        for symbol in cfg.coins:
            if _has_open_position(api, symbol):
                print(f"{symbol}: position open, skip")
                continue

            bars = _get_crypto_bars(api, symbol, limit=cfg.bars_limit)
            if bars.empty:
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
                print(f"{symbol}: KING pass ({king_conf:.2f})")
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

            ok = _buy_market(api, symbol, notional_usd=notional)
            if ok:
                print(f"{symbol}: BUY (notional ${notional:.2f})")
            else:
                print(f"{symbol}: BUY failed")

        print(f"Sleeping {cfg.poll_seconds}s...")
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
