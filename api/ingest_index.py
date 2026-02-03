
import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

# Ensure we can import from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Load env variables from root .env
dotenv_path = os.path.join(parent_dir, '.env')
load_dotenv(dotenv_path)

# Load env variables from web/.env.local (Supabase keys often here)
web_env_path = os.path.join(parent_dir, 'web', '.env.local')
if os.path.exists(web_env_path):
    load_dotenv(web_env_path, override=True)

try:
    from api.stock_ai import sync_df_to_supabase, _init_supabase
except ImportError:
    from api.stock_ai import sync_df_to_supabase, _init_supabase

def ingest_local_index(file_path, symbol="EGX30.INDX"):
    print(f"Reading {file_path} for symbol {symbol}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)
        if "date" not in df.columns:
             print("Error: JSON must have a 'date' column.")
             return

        # Ensure numeric columns
        cols = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Sync to Supabase
        print(f"Uploading {len(df)} rows to Supabase...")
        _init_supabase()
        success, msg = sync_df_to_supabase(symbol, df)
        
        if success:
            print(f"✅ Successfully ingested {symbol}: {msg}")
        else:
            print(f"❌ Failed to ingest {symbol}: {msg}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    # Check for arguments
    target_file = os.path.join(parent_dir, "symbols_data", "EGX30-INDEX.json")
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    ingest_local_index(target_file, "EGX30.INDX")
