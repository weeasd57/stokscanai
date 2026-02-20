
import sys
import os

# Add the api directory to the path
sys.path.append(os.getcwd())

from api.tradingview_integration import fetch_tradingview_prices

def test_fetch():
    symbol = "ACX/USDT.BINANCE"
    print(f"Testing fetch for {symbol}...")
    
    # Mocking sync_df_to_supabase in api.stock_ai
    import api.stock_ai as stock_ai
    original_sync = stock_ai.sync_df_to_supabase
    stock_ai.sync_df_to_supabase = lambda sym, df: (True, f"MOCKED (count={len(df)})")
    
    # Also need to mock _get_supabase_info and _last_trading_day if they fail
    original_info = stock_ai._get_supabase_info
    stock_ai._get_supabase_info = lambda sym: {"last_date": None, "count": 0}
    
    success, message = fetch_tradingview_prices(symbol, max_days=30)
    print(f"Success: {success}")
    print(f"Message: {message}")
    
    # Restore
    stock_ai.sync_df_to_supabase = original_sync
    stock_ai._get_supabase_info = original_info

if __name__ == "__main__":
    test_fetch()
