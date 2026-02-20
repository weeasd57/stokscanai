
import requests
import sys

def test_crypto_filtering():
    url = "http://localhost:8000/ai_bot/crypto_symbols_stats?timeframe=1d"
    print(f"Testing endpoint: {url}")
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Received {len(data)} symbols.")
            
            # Check for non-crypto symbols based on new filtering logic
            def is_crypto(item):
                sym = item.get("symbol", "").upper()
                return "/" in sym or ".BINANCE" in sym or sym.endswith("USD") or sym.endswith("USDT")
            
            non_crypto = [item for item in data if not is_crypto(item)]
            if non_crypto:
                print(f"Failure: Found {len(non_crypto)} symbols that don't match crypto patterns in response!")
                for item in non_crypto[:5]:
                    print(f"  - {item.get('symbol')}")
            else:
                print("Success! All symbols in response match crypto patterns.")
                if data:
                    print("Sample crypto symbols:")
                    for item in data[:5]:
                        print(f"  - {item.get('symbol')} ({item.get('exchange')})")
        else:
            print(f"Failure: {response.text}")
    except Exception as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    test_crypto_filtering()
