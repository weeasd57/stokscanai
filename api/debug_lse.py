from tvDatafeed import TvDatafeed
import time

tv = TvDatafeed()

test_symbols = ["0A05", "0A0H", "0A0L"]

print("--- Searching LSE symbols ---")
for sym in test_symbols:
    print(f"\nSearching for {sym}...")
    # Search specifically on LSE
    results = tv.search_symbol(sym, exchange="LSE")
    if results:
        print(f"Found {len(results)} results:")
        for res in results[:3]:
            print(f"  - {res['symbol']} ({res['description']}) on {res['exchange']} (type: {res['type']})")
            
            # Try fetching checking
            if res['symbol'] == sym:
                print(f"    > Attempting fetch for {sym} on {res['exchange']}...")
                try:
                    df = tv.get_hist(symbol=sym, exchange=res['exchange'], n_bars=10)
                    if df is not None and not df.empty:
                        print("    > SUCCESS: Got data!")
                    else:
                        print("    > FAIL: Empty dataframe")
                except Exception as e:
                    print(f"    > ERROR: {e}")
    else:
        print(f"No results found for {sym}")
