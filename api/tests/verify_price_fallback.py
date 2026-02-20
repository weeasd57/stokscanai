
import sys
import os

# Add the api directory to the path
sys.path.append(os.getcwd())

import api.stock_ai as stock_ai
from api.routers.admin import UpdateRequest
from fastapi import BackgroundTasks
import threading

# Mocking
stock_ai.sync_df_to_supabase = lambda sym, df: (True, f"MOCKED (count={len(df)})")
stock_ai._get_supabase_info = lambda sym: {"last_date": None, "count": 0}

# Create a mock UpdateRequest
class MockRequest:
    def __init__(self):
        self.symbols = ["ACX/USDT.BINANCE"]
        self.updatePrices = True
        self.updateFundamentals = False
        self.maxPriceDays = 30

def test_fallback():
    # We need to simulate the admin router's _price_one logic
    # instead of calling the whole endpoint which is complex to setup.
    
    from api.tradingview_integration import fetch_tradingview_prices
    from api.binance_data import fetch_binance_bars_df
    
    sym = "ACX/USDT.BINANCE"
    print(f"Testing price fetch for {sym}...")
    
    # 1. Try TV
    ok, msg = fetch_tradingview_prices(sym, max_days=30)
    print(f"TV Result: {ok}, {msg}")
    
    if not ok and sym.upper().endswith(".BINANCE"):
        print(f"Triggering fallback for {sym}...")
        bars = fetch_binance_bars_df(sym, timeframe="1d", limit=60)
        if not bars.empty:
            ok, sync_msg = stock_ai.sync_df_to_supabase(sym, bars)
            msg = f"OK (binance fallback) - {sync_msg}"
            print(f"Fallback Result: {ok}, {msg}")
        else:
            print("Fallback failed: No bars returned from Binance API.")

if __name__ == "__main__":
    test_fallback()
