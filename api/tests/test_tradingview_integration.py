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
    test_cache = "test_data_cache"
    success, msg = fetch_tradingview_prices("AAPL.US", cache_dir=test_cache, n_bars=10)
    assert success is True
    assert "OK" in msg
    
    price_path = os.path.join(test_cache, "US", "prices", "AAPL.US.csv")
    assert os.path.exists(price_path)
    
    df = pd.read_csv(price_path)
    assert not df.empty
    assert "Close" in df.columns

# @pytest.mark.skip(reason="Requires network and can be slow")
def test_fetch_tradingview_fundamentals_bulk():
    test_cache = "test_data_cache"
    symbols = ["AAPL.US", "AIR.PA"]
    out = fetch_tradingview_fundamentals_bulk(symbols, cache_dir=test_cache)
    
    assert "AAPL.US" in out
    assert "AIR.PA" in out
    
    data, meta = out["AAPL.US"]
    assert data["marketCap"] is not None
    assert meta["source"] == "tradingview"
    
    fund_path = os.path.join(test_cache, "US", "fund", "fund_AAPL.US.json")
    assert os.path.exists(fund_path)
