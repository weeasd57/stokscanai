import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, List

# Import Alpaca modules at the top level where possible
try:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest, CryptoLatestTradeRequest, StockLatestTradeRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderStatus
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    ALPACA_PY_AVAILABLE = True
except ImportError:
    ALPACA_PY_AVAILABLE = False

try:
    import alpaca_trade_api as tradeapi
    ALPACA_TRADE_API_AVAILABLE = True
except ImportError:
    ALPACA_TRADE_API_AVAILABLE = False


class _Logger(Protocol):
    def __call__(self, msg: str) -> Any: ...


class AlpacaClientError(Exception):
    """Custom exception for Alpaca client errors"""
    pass


def _looks_like_paper_url(url: str) -> bool:
    """
    Check if URL appears to be a paper trading endpoint.
    
    Args:
        url: The base URL to check
        
    Returns:
        True if URL contains 'paper', False otherwise
    """
    if not url:
        return False
    
    u = url.strip().lower()
    # More specific check for Alpaca paper trading URLs
    return 'paper' in u or 'paper-api.alpaca.markets' in u


def _parse_timeframe_for_alpaca_py(timeframe: str) -> 'TimeFrame':
    """
    Parse a timeframe string into an Alpaca TimeFrame object.
    
    Args:
        timeframe: String like "1Hour", "5Min", "1Day", etc.
        
    Returns:
        TimeFrame object
        
    Raises:
        AlpacaClientError: If alpaca-py is not available
    """
    if not ALPACA_PY_AVAILABLE:
        raise AlpacaClientError("alpaca-py is required but not installed")
    
    s = (timeframe or "").strip()
    m = re.match(r"^\s*(\d+)\s*([A-Za-z]+)\s*$", s)
    
    if not m:
        # Default to 1 hour if parse fails
        return TimeFrame(1, TimeFrameUnit.Hour)

    amount = int(m.group(1))
    unit_raw = m.group(2).lower()

    # Map string units to TimeFrameUnit enum
    unit_map = {
        'min': TimeFrameUnit.Minute,
        'mins': TimeFrameUnit.Minute,
        'minute': TimeFrameUnit.Minute,
        'minutes': TimeFrameUnit.Minute,
        'm': TimeFrameUnit.Minute,
        'hour': TimeFrameUnit.Hour,
        'hours': TimeFrameUnit.Hour,
        'h': TimeFrameUnit.Hour,
        'day': TimeFrameUnit.Day,
        'days': TimeFrameUnit.Day,
        'd': TimeFrameUnit.Day,
        'week': TimeFrameUnit.Week,
        'weeks': TimeFrameUnit.Week,
        'w': TimeFrameUnit.Week,
        'month': TimeFrameUnit.Month,
        'months': TimeFrameUnit.Month,
        'mo': TimeFrameUnit.Month,
    }
    
    unit = unit_map.get(unit_raw, TimeFrameUnit.Hour)
    return TimeFrame(amount, unit)


class AlpacaPyAdapter:
    """
    Adapter for the alpaca-py SDK providing a unified interface.
    """
    
    def __init__(
        self, 
        key_id: str, 
        secret_key: str, 
        base_url: str, 
        logger: Optional[_Logger] = None
    ):
        if not ALPACA_PY_AVAILABLE:
            raise AlpacaClientError(
                "alpaca-py is not installed. Install it with: pip install alpaca-py"
            )
        
        if not key_id or not secret_key:
            raise AlpacaClientError("API key_id and secret_key are required")
        
        paper = _looks_like_paper_url(base_url)
        
        self._logger = logger
        if self._logger:
            env_type = "!!! PAPER TRADING !!!" if paper else "=== LIVE TRADING ==="
            self._logger(f"Initializing Alpaca trading client ({env_type})")
            if paper:
                self._logger("Check your settings if you expected LIVE positions!")

        # Trading client controls positions, orders, and account
        try:
            self._trading = TradingClient(
                api_key=key_id,
                secret_key=secret_key,
                paper=paper,
                url_override=base_url or None,
                raw_data=False,
            )
        except Exception as e:
            raise AlpacaClientError(f"Failed to initialize trading client: {e}")

        # Data client is for historical market data
        try:
            self._data = CryptoHistoricalDataClient(
                api_key=key_id,
                secret_key=secret_key,
                raw_data=False,
            )
            self._stock_data = StockHistoricalDataClient(
                api_key=key_id,
                secret_key=secret_key,
                raw_data=False,
            )
        except Exception as e:
            raise AlpacaClientError(f"Failed to initialize data client: {e}")

    def get_latest_trade(self, symbol: str) -> Any:
        """
        Get the latest stock trade.
        
        Args:
            symbol: Trading symbol (e.g., "AAPL")
            
        Returns:
            Latest trade information
            
        Raises:
            AlpacaClientError: If request fails
        """
        try:
            request_params = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trade = self._stock_data.get_stock_latest_trade(request_params)
            return trade[symbol]
        except Exception as e:
            if self._logger:
                self._logger(f"Error fetching latest stock trade for {symbol}: {e}")
            raise AlpacaClientError(f"Failed to get latest stock trade: {e}")

    def get_latest_crypto_trade(self, symbol: str) -> Any:
        """
        Get the latest crypto trade.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            
        Returns:
            Latest trade information
            
        Raises:
            AlpacaClientError: If request fails
        """
        raw = (symbol or "").strip().upper()
        if not raw:
            raise AlpacaClientError("symbol is required")

        # Prefer BASE/QUOTE format (e.g. BTC/USD) for Alpaca crypto market-data requests.
        if "/" in raw:
            slash_sym = raw
        elif raw.endswith("USDT") and len(raw) > 4:
            slash_sym = f"{raw[:-4]}/USDT"
        elif raw.endswith("USD") and len(raw) > 3:
            slash_sym = f"{raw[:-3]}/USD"
        else:
            slash_sym = raw

        clean_sym = slash_sym.replace("/", "")
        attempts = [slash_sym]
        if clean_sym != slash_sym:
            attempts.append(clean_sym)

        last_err: Optional[Exception] = None
        for req_sym in attempts:
            try:
                request_params = CryptoLatestTradeRequest(symbol_or_symbols=req_sym)
                trade = self._data.get_crypto_latest_trade(request_params)
                if isinstance(trade, dict):
                    return trade.get(req_sym) or trade.get(slash_sym) or trade.get(clean_sym)
                return trade
            except Exception as e:
                last_err = e

        if self._logger:
            self._logger(f"Error fetching latest crypto trade for {symbol}: {last_err}")
        raise AlpacaClientError(f"Failed to get latest crypto trade: {last_err}")

    def list_positions(self) -> List[Any]:
        """Get all current positions."""
        try:
            return self._trading.get_all_positions()
        except Exception as e:
            if self._logger:
                self._logger(f"Error fetching positions: {e}")
            raise AlpacaClientError(f"Failed to list positions: {e}")

    def get_account(self) -> Any:
        """Get account information."""
        try:
            return self._trading.get_account()
        except Exception as e:
            if self._logger:
                self._logger(f"Error fetching account: {e}")
            raise AlpacaClientError(f"Failed to get account: {e}")

    def list_orders(self, **kwargs) -> List[Any]:
        """
        Get orders with optional status filter.
        
        Args:
            status: Order status filter ("open", "closed", "all"). Default: "open"
            
        Returns:
            List of order objects
            
        Raises:
            AlpacaClientError: If request fails
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            
            status = kwargs.get("status", "open")
            status_map = {
                "open": QueryOrderStatus.OPEN,
                "closed": QueryOrderStatus.CLOSED,
                "all": QueryOrderStatus.ALL,
            }
            status_enum = status_map.get(status, QueryOrderStatus.OPEN)
            req = GetOrdersRequest(status=status_enum)
            return self._trading.get_orders(filter=req)
        except Exception as e:
            if self._logger:
                self._logger(f"Error fetching orders: {e}")
            raise AlpacaClientError(f"Failed to list orders: {e}")

    def submit_order(self, **kwargs) -> Any:
        """
        Submit a market order.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            qty: Quantity to trade (optional if notional is provided)
            notional: Dollar amount to trade (optional if qty is provided)
            side: "buy" or "sell"
            type: Order type (only "market" supported)
            time_in_force: "gtc", "ioc", "fok", or "day" (default: "gtc")
            
        Returns:
            Order object
            
        Raises:
            AlpacaClientError: If order submission fails or invalid parameters
        """
        symbol = kwargs.get("symbol")
        qty = kwargs.get("qty")
        notional = kwargs.get("notional")
        side = (kwargs.get("side") or "").lower()
        order_type = (kwargs.get("type") or "market").lower()
        tif = (kwargs.get("time_in_force") or "gtc").lower()

        # Validation
        if not symbol:
            raise AlpacaClientError("symbol is required")
        
        if qty is None and notional is None:
            raise AlpacaClientError("Either qty or notional must be provided")
        
        if side not in {"buy", "sell"}:
            raise AlpacaClientError(f"side must be 'buy' or 'sell', got: {side}")

        if order_type != "market":
            raise AlpacaClientError(
                f"Only market orders are supported by this adapter, got type={order_type!r}"
            )

        # Convert to enums
        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
        
        tif_map = {
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
            "day": TimeInForce.DAY,
        }
        tif_enum = tif_map.get(tif)
        if tif_enum is None:
            raise AlpacaClientError(
                f"Invalid time_in_force: {tif}. Must be one of: {list(tif_map.keys())}"
            )

        # Create order request
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=float(qty) if qty is not None else None,
                notional=float(notional) if notional is not None else None,
                side=side_enum,
                type=OrderType.MARKET,
                time_in_force=tif_enum,
            )
        except (ValueError, TypeError) as e:
            raise AlpacaClientError(f"Invalid order parameters: {e}")

        # Submit order
        try:
            if self._logger:
                self._logger(f"⚡ Submitting {side.upper()} order for {symbol}...")
            return self._trading.submit_order(order_data=req)
        except Exception as e:
            if self._logger:
                self._logger(f"Order submission failed: {e}")
            raise AlpacaClientError(f"Failed to submit order: {e}")

    def get_crypto_bars(self, symbol: str, timeframe: str, limit: int) -> Any:
        """
        Get historical crypto bars.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            timeframe: Time frame string (e.g., "1Hour", "5Min")
            limit: Maximum number of bars to return
            
        Returns:
            Bars data
            
        Raises:
            AlpacaClientError: If request fails
        """
        if not symbol:
            raise AlpacaClientError("symbol is required")
        
        if limit <= 0:
            raise AlpacaClientError("limit must be positive")
        
        try:
            tf = _parse_timeframe_for_alpaca_py(timeframe)
            req = CryptoBarsRequest(
                symbol_or_symbols=symbol, 
                timeframe=tf, 
                limit=int(limit)
            )
            return self._data.get_crypto_bars(req)
        except AlpacaClientError:
            raise
        except Exception as e:
            if self._logger:
                self._logger(f"Error fetching crypto bars: {e}")
            raise AlpacaClientError(f"Failed to get crypto bars: {e}")

    def get_watchlists(self) -> List[Any]:
        """Get all watchlists."""
        try:
            return self._trading.get_watchlists()
        except Exception as e:
            if self._logger:
                self._logger(f"Error fetching watchlists: {e}")
            raise AlpacaClientError(f"Failed to get watchlists: {e}")

    def get_watchlist_by_name(self, name: str) -> Optional[Any]:
        """
        Get a watchlist by name (case-insensitive).
        
        Args:
            name: Name of the watchlist
            
        Returns:
            Watchlist object if found, None otherwise
            
        Raises:
            AlpacaClientError: If request fails
        """
        if not name:
            raise AlpacaClientError("Watchlist name is required")
        
        try:
            wls = self.get_watchlists()
            name_lower = name.lower()
            
            for wl in wls:
                if wl.name.lower() == name_lower:
                    return self._trading.get_watchlist_by_id(wl.id)
            
            if self._logger:
                self._logger(f"Watchlist '{name}' not found")
            return None
            
        except AlpacaClientError:
            raise
        except Exception as e:
            if self._logger:
                self._logger(f"Error searching for watchlist: {e}")
            raise AlpacaClientError(f"Failed to get watchlist by name: {e}")


def create_alpaca_client(
    *,
    key_id: str,
    secret_key: str,
    base_url: str,
    logger: Optional[_Logger] = None,
) -> Any:
    """
    Create an Alpaca client, trying alpaca-py first, then falling back to alpaca-trade-api.
    
    Args:
        key_id: Alpaca API key ID
        secret_key: Alpaca API secret key
        base_url: Base URL for the API
        logger: Optional logger function
        
    Returns:
        Alpaca client instance (either AlpacaPyAdapter or tradeapi.REST)
        
    Raises:
        AlpacaClientError: If neither SDK is available or initialization fails
    """
    errors = []
    
    # Try alpaca-py first (recommended)
    if ALPACA_PY_AVAILABLE:
        try:
            if logger:
                logger("Using alpaca-py SDK")
            return AlpacaPyAdapter(
                key_id=key_id, 
                secret_key=secret_key, 
                base_url=base_url, 
                logger=logger
            )
        except Exception as e:
            errors.append(f"alpaca-py initialization failed: {e}")
            if logger:
                logger(f"Failed to initialize alpaca-py: {e}")
    
    # Fall back to legacy alpaca-trade-api
    if ALPACA_TRADE_API_AVAILABLE:
        try:
            if logger:
                logger("Using legacy alpaca-trade-api SDK")
            return tradeapi.REST(key_id, secret_key, base_url, api_version="v2")
        except Exception as e:
            errors.append(f"alpaca-trade-api initialization failed: {e}")
            if logger:
                logger(f"Failed to initialize alpaca-trade-api: {e}")
    
    # Neither SDK worked
    if not ALPACA_PY_AVAILABLE and not ALPACA_TRADE_API_AVAILABLE:
        raise AlpacaClientError(
            "No Alpaca SDK available. Install one of:\n"
            "  - pip install alpaca-py (recommended)\n"
            "  - pip install alpaca-trade-api (legacy)\n"
        )
    
    # SDK available but failed to initialize
    error_msg = "Failed to initialize Alpaca client:\n" + "\n".join(f"  - {e}" for e in errors)
    raise AlpacaClientError(error_msg)
