import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client

# Workaround for directory issues
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def find_missing_logos():
    # Load env from multiple possible locations
    load_dotenv(".env")
    load_dotenv(".env.local")
    
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("Supabase credentials not found in environment variables.")
        return

    supabase: Client = create_client(url, key)
    
    # Query all symbols where logo_url is null
    response = supabase.table("stock_fundamentals").select("symbol, exchange, name").is_("logo_url", "null").execute()
    
    missing = response.data
    print(f"Found {len(missing)} symbols with missing logos.")
    
    for item in missing:
        print(f"{item['symbol']} ({item['exchange']}): {item['name']}")

if __name__ == "__main__":
    asyncio.run(find_missing_logos())
