import os
import sys
import time
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingview_integration import fetch_tradingview_fundamentals_bulk
from api.stock_ai import get_supabase_symbols
from dotenv import load_dotenv

# Load env variables
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))
load_dotenv(os.path.join(base_dir, "web", ".env.local"), override=True)

def backfill_logos(country: str = "Egypt"):
    print(f"Starting logo backfill for {country}...")
    
    # 1. Get all symbols for the country from Supabase
    symbols_data = get_supabase_symbols(country=country)
    if not symbols_data:
        print(f"No symbols found in Supabase for {country}.")
        return
    
    tickers = []
    for s in symbols_data:
        sym = s.get("symbol")
        ex = s.get("exchange")
        if sym and ex:
            tickers.append(f"{sym}.{ex}")
    
    print(f"Found {len(tickers)} symbols. Fetching logos in bulk...")
    
    # 2. Fetch fundamentals in bulk (now includes logos)
    # The function sync_data_to_supabase is called internally by fetch_tradingview_fundamentals_bulk
    results = fetch_tradingview_fundamentals_bulk(tickers)
    
    total_found = len(results)
    print(f"Backfill complete. Successfully updated {total_found} symbols.")
    
    # Log a few examples
    if total_found > 0:
        print("\nExamples:")
        for i, (ticker, (data, meta)) in enumerate(results.items()):
            if i >= 5: break
            print(f" - {ticker}: {data.get('logoUrl')}")

if __name__ == "__main__":
    backfill_logos()
