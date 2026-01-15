
import sys
import os
import json
# Add api to path
sys.path.append(os.path.join(os.getcwd(), "api"))

from tradingview_integration import fetch_tradingview_fundamentals_bulk, get_tradingview_market

from tradingview_screener import Query, Column
import pandas as pd

print("Scanning Egypt market on TradingView...")
try:
    # Fetch ALL symbols from Egypt (limit 2000 to be safe, usually less than 300)
    q = Query().set_markets("egypt")
    q = q.select("name", "description", "close", "volume", "type")
    _, df = q.limit(2000).get_scanner_data()
    
    if df is not None and not df.empty:
        print(f"Total symbols found on TradingView: {len(df)}")
        
        # User provided failing list (Base symbols)
        failing_list = [
            "AGIG", "AITG", "AIVCB", "ALRA", "AREHA", "CIRF", "EDBM", "EFMP", 
            "EKHO", "EKHOA", "EMRI", "ESGI", "ICAL", "IDHC", "IPPM", "MATD", 
            "MEDP", "MFIN", "MRCO", "NASR", "NCEM", "NCIN", "NCMP", "NOAF", 
            "ODHN", "ODID", "OREG", "ORMT", "PIOH", "REAC", "SBAG", "SMCS", 
            "SRWA", "TOUR", "UNBE", "WACE"
        ]
        
        print("\n--- Direct Match Check ---")
        tv_symbols = set(df['name'].str.upper())
        found = []
        missing = []
        for sym in failing_list:
            if sym in tv_symbols:
                found.append(sym)
            else:
                missing.append(sym)
                
        print(f"Found directly ({len(found)}): {found}")
        print(f"Still Missing ({len(missing)}): {missing}")
        
        print(f"Found directly ({len(found)}): {found}")
        print(f"Still Missing ({len(missing)}): {missing}")
        
        import difflib
        print("\n--- Suggested Aliases (Top 1 Similarity) ---")
        all_tv_names = df['name'].tolist()
        for mis in missing:
            # 1. Try to match by containing the code (e.g. EKHOA -> EKHO)
            # 2. Key word matching (hard without descriptions)
            # 3. Fuzzy match name
            matches = difflib.get_close_matches(mis, all_tv_names, n=1, cutoff=0.5)
            best_match = matches[0] if matches else "None"
            
            # Also check if missing symbol is a prefix of any TV symbol
            prefix_match = [x for x in all_tv_names if x.startswith(mis) or mis.startswith(x)]
            
            print(f"{mis:<10} -> Fuzzy: {best_match:<10} | Prefix/Contained: {prefix_match[:3]}")


except Exception as e:
    print(f"Error: {e}")


except Exception as e:
    print(f"Error: {e}")
