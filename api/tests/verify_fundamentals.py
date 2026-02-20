
import sys
import os

# Add the api directory to the path
sys.path.append(os.getcwd())

from api.tradingview_integration import get_tradingview_market, fetch_tradingview_fundamentals_bulk
from unittest.mock import MagicMock
import api.stock_ai as stock_ai

def test_fundamentals():
    symbol = "ACX/USDT.BINANCE"
    print(f"Testing fundamentals fetch for {symbol}...")
    
    market = get_tradingview_market(symbol)
    print(f"Derived market: {market}")
    
    # Mocking sync_data_to_supabase to avoid the error
    stock_ai.sync_data_to_supabase = lambda sym, data: (True, "MOCKED")
    
    results = fetch_tradingview_fundamentals_bulk([symbol])
    
    if symbol in results:
        data, meta = results[symbol]
        print(f"Success! Fetched fundamentals: {data}")
    else:
        print("Failure: Could not fetch fundamentals for ACX/USDT.BINANCE")

if __name__ == "__main__":
    test_fundamentals()
