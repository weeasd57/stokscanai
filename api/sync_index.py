
import os
import sys
import time
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from dotenv import load_dotenv

# Ensure we can import from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Load env variables from root .env (basic)
dotenv_path = os.path.join(parent_dir, '.env')
load_dotenv(dotenv_path)

# Load env variables from web/.env.local (Supabase keys often here)
web_env_path = os.path.join(parent_dir, 'web', '.env.local')
if os.path.exists(web_env_path):
    load_dotenv(web_env_path, override=True)

try:
    from api.stock_ai import sync_df_to_supabase, _init_supabase
except ImportError:
    # Fallback if running from within api folder
    from api.stock_ai import sync_df_to_supabase, _init_supabase

def sync_market_indices():
    """
    Fetches major market indices from TradingView and syncs them to Supabase
    with .INDX suffix (e.g., EGX30.INDX, GSPC.INDX) for use by the AI model.
    """
    print("Initializing Market Index Sync...")
    
    # Map Internal Symbol -> (TV Symbol, TV Exchange)
    indices = {
        "EGX30.INDX": ("EGX30", "EGX"),
        "GSPC.INDX": ("SPX", "SP"), # S&P 500 (often on SP or CBOE)
    }

    tv = TvDatafeed()
    
    for internal_sym, (tv_sym, tv_exch) in indices.items():
        print(f"Fetching {internal_sym} using {tv_sym} on {tv_exch}...")
        
        try:
            # Try fetching
            df = tv.get_hist(
                symbol=tv_sym,
                exchange=tv_exch,
                interval=Interval.in_daily,
                n_bars=5000
            )
            
            if df is None or df.empty:
                print(f"Failed to fetch {tv_sym} from {tv_exch}. Trying fallback...")
                # Fallbacks for US index
                if "SPX" in tv_sym:
                    df = tv.get_hist(symbol="SPX", exchange="CBOE", interval=Interval.in_daily, n_bars=5000)
                    if df is None or df.empty:
                        df = tv.get_hist(symbol="US500", exchange="TVC", interval=Interval.in_daily, n_bars=5000)
            
            if df is None or df.empty:
                print(f"Error: Could not fetch data for {internal_sym}")
                continue
                
            # Prepare for Supabase
            df_new = df.reset_index()
            # TV returns 'datetime' column
            df_new = df_new.rename(columns={
                'datetime': 'date', 'open': 'open', 'high': 'high', 
                'low': 'low', 'close': 'close', 'volume': 'volume'
            })
            df_new['adjusted_close'] = df_new['close']
            
            # Sync
            ok, msg = sync_df_to_supabase(internal_sym, df_new)
            if ok:
                print(f"Successfully synced {internal_sym}: {msg}")
            else:
                print(f"Failed to sync {internal_sym}: {msg}")
                
        except Exception as e:
            print(f"Exception for {internal_sym}: {e}")

if __name__ == "__main__":
    sync_market_indices()
