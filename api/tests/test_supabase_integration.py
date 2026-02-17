import os
import pytest
pytest.skip("Supabase integration test requires network and local credentials.", allow_module_level=True)

import sys
import pandas as pd
from dotenv import load_dotenv

# Add api dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(base_dir, ".env"))
load_dotenv(os.path.join(base_dir, "web", ".env.local"), override=True)

import stock_ai

def test_inference():
    print("\n--- Testing Symbol Inference ---")
    cases = [
        ("AAPL.US", ("AAPL", "US")),
        ("COMI.CC", ("COMI", "EGX")),
        ("COMI.EGX", ("COMI", "EGX")),
        ("AAPL", ("AAPL", "US")),
    ]
    for ticker, expected in cases:
        result = stock_ai._infer_symbol_exchange(ticker)
        print(f"Ticker: {ticker:10} | Result: {result} | Expected: {expected} | {'PASS' if result == expected else 'FAIL'}")

def test_supabase_read():
    print("\n--- Testing Supabase Read ---")
    stock_ai._init_supabase()
    if not stock_ai.supabase:
        print("FAIL: Supabase not initialized. Check your .env file.")
        return

    # Try a known symbol
    symbol = "COMI.EGX"
    print(f"Attempting to fetch {symbol} from Supabase...")
    try:
        df = stock_ai.get_stock_data_eodhd(None, symbol, from_date="2024-01-01", force_local=False)
        if not df.empty:
            print(f"PASS: Successfully fetched {len(df)} rows for {symbol}")
            print(df.tail(3))
        else:
            print(f"FAIL: Dataframe empty for {symbol}")
    except Exception as e:
        print(f"FAIL: Error fetching from Supabase/API: {e}")

def test_cached_tickers():
    print("\n--- Testing Cached Tickers ---")
    tickers = stock_ai.get_cached_tickers()
    print(f"Found {len(tickers)} tickers in cache (Supabase + Local)")
    # Print a few
    list_tickers = sorted(list(tickers))
    print(f"Samples: {list_tickers[:10]}")

if __name__ == "__main__":
    test_inference()
    test_supabase_read()
    test_cached_tickers()
