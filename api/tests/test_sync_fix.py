import os
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

def test_sync_case_insensitive():
    stock_ai._init_supabase()
    if not stock_ai.supabase:
        print("FAIL: Supabase not initialized.")
        return

    # 1. Test US (Lowercase)
    us_file = os.path.join(base_dir, "api", "data_cache", "US", "prices", "AAPL.US.csv")
    if os.path.exists(us_file):
        print(f"Testing US sync (lowercase headers) for {us_file}...")
        ok, msg = stock_ai.sync_local_to_supabase("AAPL.US", us_file)
        print(f"Result: {ok}, MSG: {msg}")
    else:
        print(f"SKIP: US test file not found at {us_file}")

    # 2. Test EGX (Capitalized)
    egx_file = os.path.join(base_dir, "api", "data_cache", "EGX", "prices", "COMI.EGX.csv")
    if os.path.exists(egx_file):
        print(f"\nTesting EGX sync (capitalized headers) for {egx_file}...")
        ok, msg = stock_ai.sync_local_to_supabase("COMI.EGX", egx_file)
        print(f"Result: {ok}, MSG: {msg}")
    else:
        print(f"SKIP: EGX test file not found at {egx_file}")

if __name__ == "__main__":
    test_sync_case_insensitive()
