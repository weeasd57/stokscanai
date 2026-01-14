
import sys
import os
import json
# Add api to path
sys.path.append(os.path.join(os.getcwd(), "api"))

from tradingview_integration import fetch_tradingview_fundamentals_bulk, get_tradingview_market

from tradingview_screener import Query, Column

print("Scanning Egypt market...")
try:
    # Fetch first 500 symbols from Egypt
    q = Query().set_markets("egypt")
    q = q.select("name", "description", "close", "volume")
    _, df = q.limit(500).get_scanner_data()
    
    if df is not None and not df.empty:
        print(f"Found {len(df)} symbols.")
        # Print all names
        for _, row in df.iterrows():
            name = row['name']
            desc = row['description']
            print(f"{name} | {desc}")
            
        # Specific Check
        print("\n--- Specific Check ---")
        keywords = ["Islamic", "Asyut", "Trading"]
        for kw in keywords:
            matches = df[df['description'].str.contains(kw, case=False)]
            if not matches.empty:
                print(f"\nMatches for keyword '{kw}':")
                print(matches[['name', 'description']].head(10))



except Exception as e:
    print(f"Error: {e}")
