
import requests
import sys

def test_endpoint():
    url = "http://localhost:8000/ai_bot/supabase-stats"
    print(f"Testing endpoint: {url}")
    try:
        # Note: We use port 8000 as that's where uvicorn is running in the background logs
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success! Data received:")
            print(response.json())
        else:
            print(f"Failure: {response.text}")
    except Exception as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    test_endpoint()
