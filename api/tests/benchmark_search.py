import time
import requests
import json

def benchmark_search():
    url = "http://localhost:8000/symbols/search"
    params = {
        "q": "",
        "limit": 100000,
        "country": "Egypt",
        "source": "local"
    }
    
    print(f"Starting benchmark for {url} with country=Egypt...")
    start = time.time()
    try:
        response = requests.get(url, params=params, timeout=30)
        end = time.time()
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print(f"Success! Found {len(results)} symbols.")
            print(f"Time taken: {round(end - start, 3)} seconds.")
            
            # Print first 2 results for sanity check
            if results:
                print("First 2 results:", json.dumps(results[:2], indent=2))
        else:
            print(f"Error! Status: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    benchmark_search()
