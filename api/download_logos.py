import os
import sys
import requests
from dotenv import load_dotenv
from supabase import create_client, Client

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load env variables
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))
load_dotenv(os.path.join(base_dir, "web", ".env.local"), override=True)

def download_logos(exchange: str = None, country: str = None):
    print(f"Starting local logo download for exchange={exchange}, country={country}...")
    
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("Error: Supabase credentials not found.")
        return

    supabase: Client = create_client(url, key)
    
    # Build query
    query = supabase.table("stock_fundamentals").select("symbol, data")
    if exchange:
        query = query.eq("exchange", exchange)
    if country:
        query = query.eq("country", country)
        
    response = query.execute()
    symbols_data = response.data
    
    if not symbols_data:
        print("No fundamentals found in database matching criteria.")
        return
    
    # Filter for items that have logoUrl in their data
    valid_items = []
    for item in symbols_data:
        data = item.get("data") or {}
        logo_url = data.get("logoUrl") or data.get("logo_url") # Check both
        if logo_url:
            valid_items.append({
                "symbol": item.get("symbol"),
                "logo_url": logo_url
            })
            
    print(f"Found {len(valid_items)} symbols with logo URLs in fundamental data.")
    
    # Path to save logos
    save_dir = os.path.join(base_dir, "web", "public", "logos")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    count = 0
    for item in valid_items:
        symbol = item.get("symbol")
        logo_url = item.get("logo_url")
        
        if not symbol or not logo_url:
            continue
            
        filename = f"{symbol}.svg"
        filepath = os.path.join(save_dir, filename)
        
        # Skip if already exists
        if os.path.exists(filepath):
            continue
            
        try:
            print(f"Downloading {symbol} logo...")
            resp = requests.get(logo_url, timeout=10)
            if resp.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                count += 1
            else:
                print(f"Failed to download {symbol}: {resp.status_code}")
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            
    print(f"Finished! Downloaded {count} new logos to {save_dir}.")
    return count

if __name__ == "__main__":
    download_logos()
