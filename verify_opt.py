import requests
import json

url = "http://127.0.0.1:8000/backtest/optimize"
payload = {
    "exchange": "CRYPTO",
    "model": "KING_CRYPTO 👑.pkl",
    "start_date": "2026-01-01",
    "end_date": "2026-02-01",
    "wave_values": [0.7, 0.8],
    "validator_values": [0.0, 0.6],
    "target_values": [10],
    "stoploss_values": [5],
    "capital": 100000,
    "timeframe": "1h"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
