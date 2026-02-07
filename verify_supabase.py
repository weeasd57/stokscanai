import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def verify():
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    
    print(f"Connecting to: {url}")
    if not url or not key:
        print("Error: Supabase credentials missing!")
        return

    try:
        supabase: Client = create_client(url, key)
        print("Supabase client created.")
        
        tables = ["alpaca_assets_cache", "stock_prices", "stock_bars_intraday"]
        for table in tables:
            try:
                res = supabase.table(table).select("count", count="exact").limit(1).execute()
                print(f"Table '{table}': OK. Count={res.count}")
            except Exception as e:
                print(f"Table '{table}': FAILED. Error: {e}")
                
    except Exception as e:
        print(f"Supabase connection FAILED: {e}")

if __name__ == "__main__":
    verify()
