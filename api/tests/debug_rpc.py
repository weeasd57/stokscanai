
import os
import sys
sys.path.append(os.getcwd())
from api import stock_ai
from api.stock_ai import _init_supabase

def debug_rpc():
    _init_supabase()
    if not stock_ai.supabase:
        print("Supabase not initialized")
        return
    
    try:
        # Test 1h timeframe
        timeframe = "1h"
        print(f"Calling RPC get_crypto_symbol_stats with timeframe={timeframe}...")
        res = stock_ai.supabase.rpc("get_crypto_symbol_stats", {"p_timeframe": timeframe}).execute()
        
        if res.data:
            print(f"Received {len(res.data)} items.")
            print("First item sample:")
            print(res.data[0])
            
            # Check keys
            keys = list(res.data[0].keys())
            print(f"Available keys: {keys}")
        else:
            print("No data returned from RPC.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_rpc()
