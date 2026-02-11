
import requests
import time
import os
import sys
from datetime import datetime

# Configuration
API_URL = "http://127.0.0.1:8000"
BOT_ID = "primary"
POLL_INTERVAL = 2  # seconds

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_timestamp(ts_str):
    if not ts_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ts_str

def monitor():
    print(f"Connecting to Bot Monitor at {API_URL}...")
    
    while True:
        try:
            # 1. Get Bot Status
            status_resp = requests.get(f"{API_URL}/ai_bot/status?bot_id={BOT_ID}")
            if status_resp.status_code != 200:
                print(f"Error fetching status: {status_resp.status_code}")
                time.sleep(5)
                continue
            
            status = status_resp.json()
            
            # 2. Get Recent Logs
            logs_resp = requests.get(f"{API_URL}/ai_bot/logs?bot_id={BOT_ID}&lines=10")
            logs = logs_resp.json().get("logs", []) if logs_resp.status_code == 200 else []

            clear_screen()
            
            # Header
            print("=" * 80)
            print(f"🤖 BOT RUNTIME MONITOR - {status['config'].get('name', 'Bot')} ({BOT_ID})")
            print("=" * 80)
            
            # Live Direction / Activity
            print(f"📍 CURRENT ACTIVITY: {status.get('current_activity', 'N/A')}")
            print(f"⏱️ UPTIME:           {status.get('uptime', 'N/A')}")
            print(f"🕒 LAST SCAN:       {status.get('last_scan', 'N/A')}")
            print(f"🚦 STATUS:          {status.get('status', 'N/A').upper()}")
            print(f"📈 ACTIVE POSITIONS: {len(status.get('trades', []))}") # Note: this displays trades from status, adjust if needed
            print("-" * 80)
            
            # Stats (from data_stream)
            ds = status.get("data_stream", {})
            print("📊 DATA STREAM STATUS:")
            for symbol, info in ds.items():
                stat = info.get("status", "UNKNOWN")
                color = ""
                if stat == "OK": stat = "[ OK ]"
                elif stat == "ERROR": stat = "[ERROR]"
                print(f"  {symbol:<10} | {stat:<7} | {info.get('source', 'N/A'):<10} | {format_timestamp(info.get('timestamp'))}")
            
            print("-" * 80)
            
            # Logs
            print("📜 RECENT RUNTIME LOGS:")
            for log in logs[-8:]:
                print(f"  {log}")
            
            print("=" * 80)
            print(f"Last updated: {datetime.now().strftime('%H:%M:%S')} (Polling every {POLL_INTERVAL}s)")
            print("Press Ctrl+C to stop.")
            
        except requests.exceptions.ConnectionError:
            print(f"\r❌ Error: Could not connect to API at {API_URL}. Is the server running? ", end="")
        except Exception as e:
            print(f"\r❌ Monitor Error: {e} ", end="")
            
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        sys.exit(0)
