
import requests
import json
import time

def test_hourly_sync():
    # Calling the update_batch endpoint
    url = "http://localhost:8000/admin/update_batch"
    payload = {
        "symbols": ["ALT/USDT.BINANCE"],
        "updatePrices": True,
        "updateFundamentals": False,
        "maxPriceDays": 365,
        "timeframe": "1h"
    }
    
    print(f"Sending request to {url} with timeframe=1h...")
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response Data:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    test_hourly_sync()
