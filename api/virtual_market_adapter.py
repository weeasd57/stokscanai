from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


PriceProvider = Callable[[str], Optional[float]]


def _supabase_last_close(symbol: str) -> Optional[float]:
    """Best-effort: fetch the latest close price from stock_bars_intraday."""
    try:
        from api.stock_ai import supabase, _init_supabase
        _init_supabase()
        if not supabase:
            return None
        resp = (
            supabase.table("stock_bars_intraday")
            .select("close")
            .eq("symbol", symbol)
            .order("ts", desc=True)
            .limit(1)
            .execute()
        )
        if resp and resp.data:
            val = float(resp.data[0].get("close", 0))
            if val > 0:
                return val
    except Exception:
        pass
    return None


@dataclass
class VirtualOrder:
    id: str
    client_order_id: str
    created_at: str
    updated_at: Optional[str]
    submitted_at: Optional[str]
    filled_at: Optional[str]
    order_type: str
    side: str
    time_in_force: str
    limit_price: Optional[str]
    stop_price: Optional[str]
    filled_avg_price: Optional[str]
    status: str
    symbol: str
    qty: str
    filled_qty: str


@dataclass
class VirtualPosition:
    symbol: str
    qty: str
    avg_entry_price: str
    market_value: str
    unrealized_pl: str
    unrealized_plpc: str


class _LatestTrade:
    def __init__(self, price: float):
        self.price = float(price)


class VirtualMarketAdapter:
    """
    In-memory "broker" used by LiveBot to simulate trading without Alpaca.

    Interface intentionally mirrors the small subset LiveBot + routers rely on:
    - submit_order(**kwargs)
    - close_position(symbol) -> bool
    - list_positions()
    - list_orders(status=..., limit=...)
    - get_account()
    - get_latest_crypto_trade(symbol) / get_latest_trade(symbol)
    """

    def __init__(
        self,
        *,
        initial_cash: float = 10_000.0,
        price_provider: Optional[PriceProvider] = None,
        logger: Optional[Callable[[str], Any]] = None,
    ):
        self._cash = float(initial_cash)
        self._positions: Dict[str, Dict[str, float]] = {}  # symbol -> {qty, avg_entry}
        self._orders: List[VirtualOrder] = []
        self._price_provider = price_provider
        self._log = logger or logging.getLogger(__name__).info

    def set_price_provider(self, fn: Optional[PriceProvider]) -> None:
        self._price_provider = fn

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _get_price(self, symbol: str) -> Optional[float]:
        # 1) injected provider (populated from bars the bot already fetched)
        if self._price_provider:
            try:
                p = self._price_provider(symbol)
                if p is not None and float(p) > 0:
                    return float(p)
            except Exception:
                pass

        # 2) Supabase last close (zero-network-latency if DB is warm)
        try:
            sb_price = _supabase_last_close(symbol)
            if sb_price is not None and sb_price > 0:
                return sb_price
        except Exception:
            pass

        # 3) best-effort fallback using yfinance (networked)
        try:
            import yfinance as yf

            s = (symbol or "").strip().upper()
            yf_sym = s
            if "/" in yf_sym:
                base, quote = yf_sym.split("/", 1)
                if quote in ("USD", "USDT", "USDC"):
                    yf_sym = f"{base}-USD"
                else:
                    yf_sym = f"{base}-{quote}"

            t = yf.Ticker(yf_sym)

            try:
                fi = getattr(t, "fast_info", None)
                if fi:
                    last = fi.get("last_price")
                    if last is not None and float(last) > 0:
                        return float(last)
            except Exception:
                pass

            try:
                hist = t.history(period="1d")
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    return float(hist["Close"].iloc[-1])
            except Exception:
                pass
        except Exception:
            pass

        # No price available — return None so callers can decide
        self._log(f"WARNING: No price available for {symbol}")
        return None

    def get_account(self) -> Any:
        class MockAccount:
            def __init__(self, cash: float, equity: float):
                self.cash = str(cash)
                self.equity = str(equity)
                self.status = "ACTIVE"
                self.currency = "USD"
                self.buying_power = str(cash)

        equity = float(self._cash)
        for sym, p in self._positions.items():
            qty = float(p.get("qty", 0))
            px = self._get_price(sym)
            if px is None:
                px = float(p.get("avg_entry", 0))  # fallback to entry price
            equity += qty * px
        return MockAccount(self._cash, equity)

    def list_positions(self) -> List[Any]:
        out: List[Any] = []
        for sym, p in self._positions.items():
            qty = float(p.get("qty", 0))
            avg = float(p.get("avg_entry", 0))
            px = self._get_price(sym)
            if px is None:
                px = avg  # fallback to entry price if no market data
            mv = qty * px
            upl = (px - avg) * qty
            uplpc = ((px / avg) - 1.0) if avg > 0 else 0.0
            out.append(
                VirtualPosition(
                    symbol=sym,
                    qty=str(qty),
                    avg_entry_price=str(avg),
                    market_value=str(mv),
                    unrealized_pl=str(upl),
                    unrealized_plpc=str(uplpc),
                )
            )
        return out

    def list_orders(self, **kwargs) -> List[Any]:
        status = (kwargs.get("status") or "open").lower()
        limit = kwargs.get("limit")
        try:
            limit_i = int(limit) if limit is not None else None
        except Exception:
            limit_i = None

        if status == "open":
            rows = [o for o in self._orders if (o.status or "").lower() in ("new", "open", "accepted")]
        elif status == "closed":
            rows = [o for o in self._orders if (o.status or "").lower() in ("filled", "canceled", "rejected", "done_for_day", "expired")]
        else:
            rows = list(self._orders)

        rows = list(reversed(rows))  # newest first
        return rows[:limit_i] if limit_i else rows

    def get_latest_trade(self, symbol: str) -> Any:
        return _LatestTrade(self._get_price(symbol))

    def get_latest_crypto_trade(self, symbol: str) -> Any:
        return _LatestTrade(self._get_price(symbol))

    def submit_order(self, **kwargs) -> Any:
        symbol = str(kwargs.get("symbol") or "").strip().upper()
        side = str(kwargs.get("side") or "").strip().lower()
        order_type = str(kwargs.get("type") or "market").strip().lower()
        tif = str(kwargs.get("time_in_force") or "gtc").strip().lower()
        client_order_id = str(kwargs.get("client_order_id") or f"virt_{uuid.uuid4().hex[:12]}")

        if not symbol:
            raise ValueError("symbol is required")
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got: {side!r}")
        if order_type != "market":
            raise ValueError("Virtual adapter only supports market orders")

        qty = kwargs.get("qty")
        notional = kwargs.get("notional")
        px = self._get_price(symbol)

        # Refuse to fill at unknown price — prevents phantom 1.0 trades
        if px is None or px <= 0:
            raise ValueError(f"Cannot fill order for {symbol}: no market price available")

        if qty is None and notional is None:
            raise ValueError("Either qty or notional must be provided")

        if qty is None:
            n = float(notional)
            qty_f = (n / px) if px > 0 else 0.0
        else:
            qty_f = float(qty)

        if qty_f <= 0:
            raise ValueError("qty must be > 0")

        # Apply fill immediately
        if side == "buy":
            cost = qty_f * px
            self._cash -= cost
            pos = self._positions.get(symbol)
            if pos:
                old_qty = float(pos["qty"])
                old_avg = float(pos["avg_entry"])
                new_qty = old_qty + qty_f
                new_avg = ((old_qty * old_avg) + cost) / new_qty if new_qty > 0 else px
                self._positions[symbol] = {"qty": new_qty, "avg_entry": new_avg}
            else:
                self._positions[symbol] = {"qty": qty_f, "avg_entry": px}
        else:
            pos = self._positions.get(symbol)
            if not pos:
                raise ValueError(f"No position to sell for {symbol}")
            old_qty = float(pos["qty"])
            sell_qty = min(old_qty, qty_f)
            proceeds = sell_qty * px
            self._cash += proceeds
            remaining = old_qty - sell_qty
            if remaining <= 1e-12:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol]["qty"] = remaining

        oid = str(kwargs.get("id") or f"virt_oid_{uuid.uuid4().hex[:10]}")
        now = self._now()
        order = VirtualOrder(
            id=oid,
            client_order_id=client_order_id,
            created_at=now,
            updated_at=now,
            submitted_at=now,
            filled_at=now,
            order_type=order_type,
            side=side,
            time_in_force=tif,
            limit_price=None,
            stop_price=None,
            filled_avg_price=str(px),
            status="filled",
            symbol=symbol,
            qty=str(qty_f),
            filled_qty=str(qty_f),
        )
        self._orders.append(order)
        self._log(f"VIRTUAL ORDER filled: {side.upper()} {qty_f} {symbol} @ {px}")
        return order

    def close_position(self, symbol: str) -> bool:
        sym = str(symbol or "").strip().upper()
        pos = self._positions.get(sym)
        if not pos:
            return False

        qty_f = float(pos.get("qty", 0))
        if qty_f <= 0:
            self._positions.pop(sym, None)
            return False

        # Full close via a market sell.
        self.submit_order(symbol=sym, qty=qty_f, side="sell", type="market", time_in_force="gtc")
        return True


def create_virtual_market_client(*, logger: Optional[Callable[[str], Any]] = None) -> VirtualMarketAdapter:
    return VirtualMarketAdapter(logger=logger)

