import os
import sys
import time
import warnings
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class TradeAction(Enum):
    """أنواع الإجراءات التداولية"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ExitReason(Enum):
    """أسباب الخروج من الصفقة"""
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_EXIT = "TIME_EXIT"
    MANUAL = "MANUAL"
    EMERGENCY = "EMERGENCY"


@dataclass(frozen=True)
class BotConfig:
    """إعدادات البوت الشاملة"""
    # Alpaca API
    alpaca_key_id: str
    alpaca_secret_key: str
    alpaca_base_url: str
    
    # قائمة العملات
    coins: list[str]
    
    # عتبات النماذج
    king_threshold: float
    council_threshold: float
    
    # إدارة رأس المال
    max_notional_usd: float
    pct_cash_per_trade: float
    max_total_exposure_usd: float  # جديد: حد التعرض الإجمالي
    max_positions: int  # جديد: حد عدد الصفقات المفتوحة
    min_trade_size_usd: float  # جديد: الحد الأدنى لحجم الصفقة
    
    # إعدادات البيانات
    bars_limit: int
    poll_seconds: int
    timeframe: str
    
    # إدارة المخاطر - الخروج
    enable_sells: bool
    target_pct: float
    stop_loss_pct: float
    hold_max_bars: int
    
    # وقف الخسارة المتحرك
    use_trailing: bool
    trail_be_pct: float
    trail_lock_trigger_pct: float
    trail_lock_pct: float
    
    # جديد: حماية متقدمة
    max_daily_trades: int  # الحد الأقصى للصفقات اليومية
    max_daily_loss_pct: float  # الحد الأقصى للخسارة اليومية
    circuit_breaker_loss_pct: float  # إيقاف التداول عند خسارة معينة
    require_complete_bars: bool  # انتظار اكتمال الشمعة
    
    # جديد: فلاتر إضافية
    min_volume_usd: float  # الحد الأدنى لحجم التداول
    max_spread_pct: float  # الحد الأقصى للفارق السعري
    volatility_filter: bool  # تفعيل فلتر التقلبات
    max_atr_multiplier: float  # مضاعف ATR للتقلبات
    
    # المسارات
    state_path: str
    trades_log_path: str
    performance_log_path: str
    alerts_log_path: str


@dataclass
class Position:
    """معلومات الصفقة المفتوحة"""
    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: float
    current_stop: float
    bars_held: int
    trail_mode: str
    highest_price: float
    initial_stop: float
    notional_usd: float


@dataclass
class DailyStats:
    """إحصائيات التداول اليومية"""
    date: str
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    starting_balance: float = 0.0
    current_balance: float = 0.0
    max_drawdown: float = 0.0


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """إعداد نظام السجلات المتقدم"""
    Path(log_dir).mkdir(exist_ok=True)
    
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.DEBUG)
    
    # تنسيق السجلات
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # معالج للملف (جميع المستويات)
    fh = logging.FileHandler(
        os.path.join(log_dir, f"bot_{datetime.now().strftime('%Y%m%d')}.log"),
        encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    
    # معالج للشاشة (INFO فما فوق)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# CONFIGURATION BUILDER
# ============================================================================

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


def _parse_int(value: str, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _parse_coins(raw: str) -> list[str]:
    coins = []
    for part in (raw or "").split(","):
        s = part.strip().upper()
        if not s:
            continue
        coins.append(s)
    return coins


def _build_config() -> BotConfig:
    """بناء الإعدادات المتقدمة"""
    coins = _parse_coins(
        _read_env("LIVE_COINS", "BTC/USD,ETH/USD,LTC/USD,DOGE/USD")
    )

    key_id = _read_env("ALPACA_KEY_ID", None) or _read_env("ALPACA_API_KEY", None)
    secret = _read_env("ALPACA_SECRET_KEY", None)
    if not key_id:
        raise RuntimeError("Missing required env var: ALPACA_KEY_ID")
    if not secret:
        raise RuntimeError("Missing required env var: ALPACA_SECRET_KEY")

    return BotConfig(
        # API
        alpaca_key_id=str(key_id),
        alpaca_secret_key=str(secret),
        alpaca_base_url=str(_read_env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")),
        
        # العملات
        coins=coins,
        
        # عتبات النماذج
        king_threshold=_parse_float(_read_env("KING_THRESHOLD", "0.60"), 0.60),
        council_threshold=_parse_float(_read_env("COUNCIL_THRESHOLD", "0.35"), 0.35),
        
        # إدارة رأس المال - محسّن
        max_notional_usd=_parse_float(_read_env("MAX_NOTIONAL_USD", "500"), 500.0),
        pct_cash_per_trade=_parse_float(_read_env("PCT_CASH_PER_TRADE", "0.10"), 0.10),
        max_total_exposure_usd=_parse_float(_read_env("MAX_TOTAL_EXPOSURE_USD", "2000"), 2000.0),
        max_positions=_parse_int(_read_env("MAX_POSITIONS", "5"), 5),
        min_trade_size_usd=_parse_float(_read_env("MIN_TRADE_SIZE_USD", "10"), 10.0),
        
        # البيانات
        bars_limit=_parse_int(_read_env("BARS_LIMIT", "200"), 200),
        poll_seconds=_parse_int(_read_env("POLL_SECONDS", "300"), 300),
        timeframe=str(_read_env("TIMEFRAME", "1Hour") or "1Hour"),
        
        # إدارة المخاطر الأساسية
        enable_sells=_parse_bool(_read_env("LIVE_ENABLE_SELLS", "1"), True),
        target_pct=_parse_float(_read_env("LIVE_TARGET_PCT", "0.10"), 0.10),
        stop_loss_pct=_parse_float(_read_env("LIVE_STOP_LOSS_PCT", "0.05"), 0.05),
        hold_max_bars=_parse_int(_read_env("LIVE_HOLD_MAX_BARS", "20"), 20),
        
        # الوقف المتحرك
        use_trailing=_parse_bool(_read_env("LIVE_USE_TRAILING", "1"), True),
        trail_be_pct=_parse_float(_read_env("LIVE_TRAIL_BE_PCT", "0.05"), 0.05),
        trail_lock_trigger_pct=_parse_float(_read_env("LIVE_TRAIL_LOCK_TRIGGER_PCT", "0.08"), 0.08),
        trail_lock_pct=_parse_float(_read_env("LIVE_TRAIL_LOCK_PCT", "0.05"), 0.05),
        
        # حماية متقدمة - جديد
        max_daily_trades=_parse_int(_read_env("MAX_DAILY_TRADES", "20"), 20),
        max_daily_loss_pct=_parse_float(_read_env("MAX_DAILY_LOSS_PCT", "0.10"), 0.10),
        circuit_breaker_loss_pct=_parse_float(_read_env("CIRCUIT_BREAKER_LOSS_PCT", "0.15"), 0.15),
        require_complete_bars=_parse_bool(_read_env("REQUIRE_COMPLETE_BARS", "1"), True),
        
        # فلاتر إضافية - جديد
        min_volume_usd=_parse_float(_read_env("MIN_VOLUME_USD", "100000"), 100000.0),
        max_spread_pct=_parse_float(_read_env("MAX_SPREAD_PCT", "0.005"), 0.005),
        volatility_filter=_parse_bool(_read_env("VOLATILITY_FILTER", "1"), True),
        max_atr_multiplier=_parse_float(_read_env("MAX_ATR_MULTIPLIER", "3.0"), 3.0),
        
        # المسارات
        state_path=str(_read_env("LIVE_STATE_PATH", "state/bot_state.json")),
        trades_log_path=str(_read_env("TRADES_LOG_PATH", "logs/trades.json")),
        performance_log_path=str(_read_env("PERFORMANCE_LOG_PATH", "logs/performance.json")),
        alerts_log_path=str(_read_env("ALERTS_LOG_PATH", "logs/alerts.json")),
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _normalize_symbol(symbol: str) -> str:
    """تطبيع رمز العملة"""
    return (symbol or "").strip().upper().replace("/", "")


def _format_symbol_for_bot(symbol: str) -> str:
    """تنسيق رمز العملة للبوت (BTC/USD)"""
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


def _load_json(path: str) -> dict:
    """تحميل ملف JSON"""
    try:
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_json(path: str, data: dict) -> None:
    """حفظ ملف JSON بشكل آمن"""
    try:
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        logging.error(f"Failed to save {path}: {e}")


def _is_bar_complete(timestamp: pd.Timestamp, timeframe: str) -> bool:
    """
    التحقق من اكتمال الشمعة
    يمنع التداول على شمعة غير مكتملة
    """
    try:
        now = pd.Timestamp.now(tz='UTC')
        
        # تحويل الإطار الزمني إلى دقائق
        if "Min" in timeframe or "min" in timeframe:
            minutes = int(''.join(filter(str.isdigit, timeframe)))
        elif "Hour" in timeframe or "hour" in timeframe:
            minutes = int(''.join(filter(str.isdigit, timeframe))) * 60
        elif "Day" in timeframe or "day" in timeframe:
            minutes = 1440
        else:
            minutes = 60  # افتراضي: ساعة واحدة
        
        # حساب وقت إغلاق الشمعة المتوقع
        bar_start = timestamp.floor(f'{minutes}min')
        bar_end = bar_start + pd.Timedelta(minutes=minutes)
        
        # الشمعة مكتملة إذا مر وقت إغلاقها
        return now >= bar_end
    except Exception:
        return False


# ============================================================================
# MODEL LOADING
# ============================================================================

def _load_models(logger: logging.Logger):
    """تحميل نماذج التعلم الآلي"""
    try:
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

        logger.info("✓ Models loaded successfully")
        return king_obj, king_clf, validator
    except Exception as e:
        logger.error(f"✗ Model load failed: {e}")
        raise


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def _align_for_king(X_src: pd.DataFrame, king_artifact: object) -> pd.DataFrame:
    """محاذاة الميزات لنموذج KING"""
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


def _prepare_features(bars: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """تحضير الميزات الفنية"""
    try:
        from api.train_exchange_model import (
            add_technical_indicators,
            add_indicator_signals,
            add_massive_features
        )

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
    except Exception as e:
        logger.error(f"Feature preparation failed: {e}")
        return pd.DataFrame()


# ============================================================================
# ALPACA API WRAPPERS
# ============================================================================

def _get_crypto_bars(api, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """جلب بيانات الشموع"""
    try:
        bars = api.get_crypto_bars(symbol, timeframe=str(timeframe or "1Hour"), limit=int(limit)).df
        if bars is None or getattr(bars, "empty", True):
            return pd.DataFrame()
        out = bars.reset_index()
        logger.debug(f"{symbol}: Retrieved {len(out)} bars")
        return out
    except Exception as e:
        logger.error(f"{symbol}: Failed to get bars: {e}")
        return pd.DataFrame()


def _get_account_info(api, logger: logging.Logger) -> Tuple[float, float]:
    """الحصول على معلومات الحساب (الرصيد، قيمة المحفظة)"""
    try:
        account = api.get_account()
        cash = float(getattr(account, "cash", 0) or 0)
        equity = float(getattr(account, "equity", 0) or 0)
        return cash, equity
    except Exception as e:
        logger.error(f"Failed to get account info: {e}")
        return 0.0, 0.0


def _get_open_positions(api, logger: logging.Logger) -> List:
    """الحصول على جميع الصفقات المفتوحة"""
    try:
        return api.list_positions()
    except Exception as e:
        logger.error(f"Failed to list positions: {e}")
        return []


def _get_position_by_symbol(api, symbol: str, logger: logging.Logger):
    """البحث عن صفقة مفتوحة لعملة معينة"""
    try:
        norm = _normalize_symbol(symbol)
        for p in _get_open_positions(api, logger):
            if _normalize_symbol(getattr(p, "symbol", "")) == norm:
                return p
        return None
    except Exception:
        return None


def _calculate_total_exposure(api, logger: logging.Logger) -> float:
    """حساب إجمالي التعرض المالي"""
    total = 0.0
    try:
        for p in _get_open_positions(api, logger):
            market_value = float(getattr(p, "market_value", 0) or 0)
            total += abs(market_value)
    except Exception as e:
        logger.error(f"Failed to calculate exposure: {e}")
    return total


def _buy_market(api, symbol: str, notional_usd: float, logger: logging.Logger):
    """تنفيذ أمر شراء"""
    try:
        order = api.submit_order(
            symbol=symbol,
            notional=float(notional_usd),
            side="buy",
            type="market",
            time_in_force="gtc",
        )
        logger.info(f"✓ {symbol}: BUY order submitted (${notional_usd:.2f})")
        return order
    except Exception as e:
        logger.error(f"✗ {symbol}: BUY order failed: {e}")
        return None


def _sell_market(api, symbol: str, qty: float, logger: logging.Logger):
    """تنفيذ أمر بيع"""
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=float(qty),
            side="sell",
            type="market",
            time_in_force="gtc",
        )
        logger.info(f"✓ {symbol}: SELL order submitted (qty={qty:.8f})")
        return order
    except Exception as e:
        logger.error(f"✗ {symbol}: SELL order failed: {e}")
        return None


def _include_open_positions_in_coins(api, coins: list[str], logger: logging.Logger) -> list[str]:
    """
    Ensure any existing open positions are managed by the bot even if not listed in LIVE_COINS.
    """
    try:
        have = {_normalize_symbol(c) for c in (coins or [])}
        out = list(coins or [])
        for p in _get_open_positions(api, logger):
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


def _maybe_sell_position(*, api, cfg: BotConfig, symbol: str, pos, bars: pd.DataFrame, state: dict, logger: logging.Logger) -> bool:
    """
    Backtest-style exits on the latest bar with advanced trailing stop
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
    highest_price = sym_state.get("highest_price", entry_price)

    # تحديث أعلى سعر
    if hi > highest_price:
        highest_price = hi
        sym_state["highest_price"] = highest_price

    # Conservative bar evaluation using current stop/target
    if lo <= float(current_stop):
        try:
            qty = float(getattr(pos, "qty", 0) or 0)
        except Exception:
            qty = 0.0
        if qty > 0:
            order = _sell_market(api, symbol, qty=qty, logger=logger)
            if order is not None:
                pnl_pct = ((current_stop - entry_price) / entry_price) if entry_price > 0 else 0
                logger.info(f"{symbol}: SELL (STOP) qty={qty} stop=${float(current_stop):.6f} entry=${float(entry_price):.6f} PnL={pnl_pct:+.2%}")
                state.pop(key, None)
                return True
        return False

    if hi >= float(take_profit):
        try:
            qty = float(getattr(pos, "qty", 0) or 0)
        except Exception:
            qty = 0.0
        if qty > 0:
            order = _sell_market(api, symbol, qty=qty, logger=logger)
            if order is not None:
                pnl_pct = ((take_profit - entry_price) / entry_price) if entry_price > 0 else 0
                logger.info(f"{symbol}: SELL (TARGET) qty={qty} tp=${float(take_profit):.6f} entry=${float(entry_price):.6f} PnL={pnl_pct:+.2%}")
                state.pop(key, None)
                return True
        return False

    # Trailing stop updates (effective next bar)
    if cfg.use_trailing:
        be_price = float(entry_price)
        lock_price = float(entry_price) * (1 + float(cfg.trail_lock_pct))
        lock_trigger = float(entry_price) * (1 + float(cfg.trail_lock_trigger_pct))
        
        if hi >= lock_trigger and float(current_stop) < lock_price:
            current_stop = lock_price
            trail_mode = "LOCK"
            logger.info(f"🔒 {symbol}: Profit locked @ ${lock_price:.6f}")
        elif hi >= float(entry_price) * (1 + float(cfg.trail_be_pct)) and float(current_stop) < be_price:
            current_stop = be_price
            trail_mode = "BE"
            logger.info(f"⚖️ {symbol}: Break-even stop @ ${be_price:.6f}")

    bars_held = int(sym_state.get("bars_held") or 0) + 1
    if bars_held >= int(cfg.hold_max_bars):
        try:
            qty = float(getattr(pos, "qty", 0) or 0)
        except Exception:
            qty = 0.0
        if qty > 0:
            order = _sell_market(api, symbol, qty=qty, logger=logger)
            if order is not None:
                pnl_pct = ((close - entry_price) / entry_price) if entry_price > 0 else 0
                logger.info(f"{symbol}: SELL (TIME) qty={qty} bars={bars_held} close=${close:.6f} PnL={pnl_pct:+.2%}")
                state.pop(key, None)
                return True

    state[key] = {
        **sym_state,
        "entry_price": float(entry_price),
        "bars_held": int(bars_held),
        "current_stop": float(current_stop),
        "trail_mode": str(trail_mode),
        "highest_price": float(highest_price),
    }
    return False


def main() -> int:
    cfg = _build_config()
    logger = setup_logging()
    state = _load_json(cfg.state_path)

    logger.info("="*70)
    logger.info("🚀 ADVANCED CRYPTO TRADING BOT STARTED")
    logger.info("="*70)
    logger.info(f"Coins: {', '.join(cfg.coins)}")
    logger.info(f"Thresholds: KING ≥ {cfg.king_threshold:.2f}, COUNCIL ≥ {cfg.council_threshold:.2f}")
    logger.info(f"Risk: Max positions={cfg.max_positions}, Max exposure=${cfg.max_total_exposure_usd:.0f}")
    logger.info(f"Polling: {cfg.poll_seconds}s | Timeframe: {cfg.timeframe}")
    if cfg.enable_sells:
        logger.info(
            f"Sells enabled: TP={cfg.target_pct:.2%} SL={cfg.stop_loss_pct:.2%} "
            f"hold_max_bars={cfg.hold_max_bars} trailing={cfg.use_trailing}"
        )
    logger.info("="*70)

    try:
        from api.alpaca_adapter import create_alpaca_client
    except Exception as e:
        logger.error(f"Failed to import Alpaca client adapter: {e}")
        return 2

    try:
        king_obj, king_clf, validator = _load_models(logger)
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return 2

    api = create_alpaca_client(
        key_id=cfg.alpaca_key_id,
        secret_key=cfg.alpaca_secret_key,
        base_url=cfg.alpaca_base_url,
        logger=logger.info,
    )

    iteration = 0
    daily_trades = 0
    daily_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    while True:
        iteration += 1
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Reset daily counter
        if current_date != daily_date:
            daily_trades = 0
            daily_date = current_date
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 Scan #{iteration} @ {now}")
        logger.info(f"{'='*70}")

        # Get account info
        cash, equity = _get_account_info(api, logger)
        positions = _get_open_positions(api, logger)
        exposure = _calculate_total_exposure(api, logger)
        
        logger.info(
            f"💰 Account: Cash=${cash:.2f} | Equity=${equity:.2f} | "
            f"Positions={len(positions)} | Exposure=${exposure:.2f} | Daily trades={daily_trades}/{cfg.max_daily_trades}"
        )

        coins = _include_open_positions_in_coins(api, cfg.coins, logger)
        for symbol in coins:
            bars = _get_crypto_bars(api, symbol, timeframe=cfg.timeframe, limit=cfg.bars_limit, logger=logger)
            if bars.empty:
                continue

            pos = _get_position_by_symbol(api, symbol, logger)
            if pos is not None:
                sold = _maybe_sell_position(api=api, cfg=cfg, symbol=symbol, pos=pos, bars=bars, state=state, logger=logger)
                _save_json(cfg.state_path, state)
                if sold:
                    continue
                logger.debug(f"{symbol}: position open, holding")
                continue

            # Check risk limits before buying
            if len(positions) >= cfg.max_positions:
                logger.debug(f"{symbol}: Max positions reached ({len(positions)}/{cfg.max_positions})")
                continue
            
            if exposure >= cfg.max_total_exposure_usd:
                logger.debug(f"{symbol}: Max exposure reached (${exposure:.2f}/${cfg.max_total_exposure_usd:.2f})")
                continue
            
            if daily_trades >= cfg.max_daily_trades:
                logger.debug(f"{symbol}: Max daily trades reached ({daily_trades}/{cfg.max_daily_trades})")
                continue

            features = _prepare_features(bars, logger)
            if features.empty:
                continue

            X_all = features.iloc[[-1]].copy()
            Xk = _align_for_king(X_all, king_obj)

            try:
                king_conf = float(king_clf.predict_proba(Xk)[:, 1][0])
            except Exception:
                continue

            if king_conf < cfg.king_threshold:
                logger.debug(f"{symbol}: KING fail ({king_conf:.2f})")
                continue

            try:
                council_conf = float(validator.predict_proba(X_all, primary_conf=np.asarray([king_conf]))[:, 1][0])
            except Exception:
                continue

            logger.info(f"{symbol}: KING={king_conf:.2f} COUNCIL={council_conf:.2f}")

            if council_conf < cfg.council_threshold:
                continue

            notional = min(cash * cfg.pct_cash_per_trade, cfg.max_notional_usd)
            if notional < cfg.min_trade_size_usd:
                logger.warning(f"{symbol}: insufficient cash ({cash:.2f})")
                continue

            order = _buy_market(api, symbol, notional_usd=notional, logger=logger)
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
                    "highest_price": avg_fill,
                }
                _save_json(cfg.state_path, state)
                daily_trades += 1
                logger.info(f"{symbol}: BUY (notional ${notional:.2f})")
            else:
                logger.warning(f"{symbol}: BUY failed")

        logger.info(f"✓ Scan completed. Sleeping {cfg.poll_seconds}s...")
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
