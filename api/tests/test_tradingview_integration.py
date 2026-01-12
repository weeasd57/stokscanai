import os
import sys
import pandas as pd
import pytest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingview_integration import (
    get_tradingview_market,
    get_tradingview_exchange,
    fetch_tradingview_prices,
    fetch_tradingview_fundamentals_bulk
)

def test_get_tradingview_market():
    assert get_tradingview_market("AAPL.US") == "america"
    assert get_tradingview_market("AIR.PA") == "france"
    assert get_tradingview_market("COMI.EGX") == "egypt"
    assert get_tradingview_market("VOD.LSE") == "uk"

def test_get_tradingview_exchange():
    assert get_tradingview_exchange("AAPL.US") == "NASDAQ"
    assert get_tradingview_exchange("AIR.PA") == "EURONEXT"
    assert get_tradingview_exchange("COMI.EGX") == "EGX"

# @pytest.mark.skip(reason="Requires network and can be slow")
def test_fetch_tradingview_prices():
    success, msg = fetch_tradingview_prices("AAPL.US", n_bars=10)
    assert success is True
    assert "OK" in msg or "TV Sync" in msg
    
# @pytest.mark.skip(reason="Requires network and can be slow")
def test_fetch_tradingview_fundamentals_bulk():
    symbols = ["AAPL.US", "AIR.PA"]
    out = fetch_tradingview_fundamentals_bulk(symbols)
    
    assert "AAPL.US" in out
    assert "AIR.PA" in out
    
    data, meta = out["AAPL.US"]
    assert data["marketCap"] is not None
    assert meta["source"] == "tradingview"
