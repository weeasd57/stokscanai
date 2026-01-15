from tradingview_screener import Query, col
import pandas as pd

def test_logos():
    print("Testing TradingView screener for logo columns...")
    
    # Try selecting 'logoid'
    try:
        q = (
            Query()
            .set_markets('egypt')
            .select('name', 'description', 'logoid')
            .limit(10)
        )
        _, df = q.get_scanner_data()
        print("\nResults for 'logoid':")
        print(df)
    except Exception as e:
        print(f"\nFailed to select 'logoid': {e}")

    # Try selecting 'logoid_company'
    try:
        q = (
            Query()
            .set_markets('egypt')
            .select('name', 'description', 'logoid_company')
            .limit(10)
        )
        _, df = q.get_scanner_data()
        print("\nResults for 'logoid_company':")
        print(df)
    except Exception as e:
        print(f"\nFailed to select 'logoid_company': {e}")

if __name__ == "__main__":
    test_logos()
