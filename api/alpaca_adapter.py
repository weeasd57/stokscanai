import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol


class _Logger(Protocol):
    def __call__(self, msg: str) -> Any: ...


def _looks_like_paper_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return "paper" in u


def _parse_timeframe_for_alpaca_py(timeframe: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    s = (timeframe or "").strip()
    m = re.match(r"^\s*(\d+)\s*([A-Za-z]+)\s*$", s)
    if not m:
        return TimeFrame(1, TimeFrameUnit.Hour)

    amount = int(m.group(1))
    unit_raw = m.group(2).lower()

    if unit_raw in {"min", "mins", "minute", "minutes", "m"}:
        unit = TimeFrameUnit.Minute
    elif unit_raw in {"hour", "hours", "h"}:
        unit = TimeFrameUnit.Hour
    elif unit_raw in {"day", "days", "d"}:
        unit = TimeFrameUnit.Day
    elif unit_raw in {"week", "weeks", "w"}:
        unit = TimeFrameUnit.Week
    elif unit_raw in {"month", "months", "mo"}:
        unit = TimeFrameUnit.Month
    else:
        unit = TimeFrameUnit.Hour

    return TimeFrame(amount, unit)


@dataclass
class _DFWrap:
    df: Any


class AlpacaPyAdapter:
    def __init__(self, key_id: str, secret_key: str, base_url: str, logger: Optional[_Logger] = None):
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.trading.client import TradingClient

        paper = _looks_like_paper_url(base_url)

        # Trading client controls positions, orders, and account.
        self._trading = TradingClient(
            api_key=key_id,
            secret_key=secret_key,
            paper=paper,
            url_override=base_url or None,
            raw_data=False,
        )

        # Data client is for historical market data (crypto bars, etc.).
        self._data = CryptoHistoricalDataClient(
            api_key=key_id,
            secret_key=secret_key,
            raw_data=False,
        )

        self._logger = logger

    def list_positions(self):
        return self._trading.get_all_positions()

    def get_account(self):
        return self._trading.get_account()

    def submit_order(self, **kwargs):
        from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        symbol = kwargs.get("symbol")
        qty = kwargs.get("qty")
        notional = kwargs.get("notional")
        side = (kwargs.get("side") or "").lower()
        order_type = (kwargs.get("type") or "").lower()
        tif = (kwargs.get("time_in_force") or "").lower()

        if order_type not in {"market", ""}:
            raise ValueError(f"Only market orders are supported by this bot, got type={order_type!r}")

        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
        tif_enum = {
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
            "day": TimeInForce.DAY,
        }.get(tif, TimeInForce.GTC)

        req = MarketOrderRequest(
            symbol=symbol,
            qty=float(qty) if qty is not None else None,
            notional=float(notional) if notional is not None else None,
            side=side_enum,
            type=OrderType.MARKET,
            time_in_force=tif_enum,
        )
        return self._trading.submit_order(order_data=req)

    def get_crypto_bars(self, symbol: str, timeframe: str, limit: int):
        from alpaca.data.requests import CryptoBarsRequest

        tf = _parse_timeframe_for_alpaca_py(timeframe)
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=int(limit))
        return self._data.get_crypto_bars(req)


def create_alpaca_client(
    *,
    key_id: str,
    secret_key: str,
    base_url: str,
    logger: Optional[_Logger] = None,
):
    try:
        import alpaca_trade_api as tradeapi  # type: ignore

        if logger:
            logger("Using legacy alpaca_trade_api SDK.")
        return tradeapi.REST(key_id, secret_key, base_url, api_version="v2")
    except Exception:
        pass

    try:
        if logger:
            logger("Using alpaca-py SDK.")
        return AlpacaPyAdapter(key_id=key_id, secret_key=secret_key, base_url=base_url, logger=logger)
    except Exception as e:
        raise ImportError(
            "Alpaca SDK not available. Install one of:\n"
            "  - pip install alpaca-py\n"
            "  - pip install alpaca-trade-api\n"
            f"Original error: {e}"
        )
