from tvDatafeed import TvDatafeed

tv = TvDatafeed()

symbols_to_test = [
    ("0A0H", "LSE"),
    ("0A0H", "LSEIOB"), # Guess
    ("0A0L", "LSE"),
    ("0A05", "LSE"), # Known good
]

print("--- Testing Direct Fetch ---")

for sym, exch in symbols_to_test:
    print(f"\nFetching {sym} on {exch}...")
    try:
        df = tv.get_hist(symbol=sym, exchange=exch, n_bars=10)
        if df is not None and not df.empty:
            print(f"SUCCESS: Got {len(df)} rows")
            print(df.tail(2))
        else:
            print("FAIL: Empty/None")
    except Exception as e:
        print(f"ERROR: {e}")
