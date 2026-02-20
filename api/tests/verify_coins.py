import requests

def test_available_coins():
    url = "http://127.0.0.1:8000/api/ai_bot/available_coins?source=virtual&pair_type=USDT"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"Count: {len(data)}")
        print(f"First 5: {data[:5]}")
        # Check if first one has .BINANCE
        if data and ".BINANCE" in data[0]:
            print("SUCCESS: Symbols contain .BINANCE suffix")
        else:
            print("FAILURE: Symbols missing .BINANCE suffix")
            print(f"Sample: {data[0] if data else 'None'}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_available_coins()
