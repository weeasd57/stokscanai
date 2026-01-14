import os
import sys
import argparse
import time
from datetime import datetime
from api.smart_sync import get_smart_sync
from api.stock_ai import _init_supabase, supabase

def run_smart_update(exchange: str, days: int, update_prices: bool, update_funds: bool, unified: bool):
    print(f"--- Smart Update Started: {datetime.now()} ---")
    print(f"Exchange: {exchange}")
    print(f"Days: {days}")
    print(f"Prices: {update_prices}, Funds: {update_funds}, Unified: {unified}")
    
    _init_supabase()
    if not supabase:
        print("Error: Supabase not initialized")
        return

    # 1. Fetch symbols for the exchange
    # We can fetch from stock_prices to see what we already have
    res = supabase.table("stock_prices").select("symbol").eq("exchange", exchange).execute()
    symbols = sorted(list(set(r["symbol"] for r in res.data)))
    
    if not symbols:
        print(f"No symbols found in cloud for exchange {exchange}. Cannot update.")
        return
        
    print(f"Found {len(symbols)} symbols to update.")
    
    syncer = get_smart_sync()
    
    # 2. Update Prices if requested
    if update_prices:
        print("Starting Prices Update...")
        result = syncer.sync_exchange_prices(exchange, symbols, max_days=days, unified_dates=unified)
        print(f"Prices Update Finished. Success: {result['success']}/{result['total']}")

    # 3. Update Fundamentals if requested
    if update_funds:
        print("Starting Fundamentals Update...")
        from api.tradingview_integration import fetch_tradingview_fundamentals_bulk
        
        # We'll use 1s throttle for fundamentals too
        chunk_size = 50 
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            print(f"Fetching fundamentals chunk {i//chunk_size + 1}...")
            fetch_tradingview_fundamentals_bulk(chunk)
            time.sleep(1) # Throttling
        print("Fundamentals Update Finished.")

    if unified:
        print("Post-processing: Enforcing unified dates...")
        syncer.enforce_unified_dates(exchange, target_days=days)

    print(f"--- Smart Update Finished: {datetime.now()} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--prices", action="store_true", default=True)
    parser.add_argument("--funds", action="store_true", default=False)
    parser.add_argument("--unified", action="store_true", default=False)
    
    args = parser.parse_args()
    
    run_smart_update(
        args.exchange, 
        args.days, 
        args.prices, 
        args.funds, 
        args.unified
    )
