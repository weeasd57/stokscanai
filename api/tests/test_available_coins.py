import requests
import json

def test_available_coins():
    url = "http://localhost:8000/ai_bot/available_coins?source=virtual&limit=100&pair_type=USDT"
    print(f"Testing URL: {url}")
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Returned {len(data)} coins.")
        if data:
            print(f"First 10 coins: {data[:10]}")
        else:
            print("Received empty list!")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Note: Ensure the API server is running (py -m uvicorn api.main:app).")

if __name__ == "__main__":
    test_available_coins()
