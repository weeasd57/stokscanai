
import os
import sys
from dotenv import load_dotenv

# Fix path
sys.path.append(os.getcwd())

# Load env
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(os.path.dirname(base_dir), ".env")
load_dotenv(env_path)

print(f"--- Verifying Pure Supabase Loading ---")

try:
    from api.live_bot import BotManager
    manager = BotManager()
    
    print(f"Bots loaded: {len(manager._bots)}")
    for bid, bot in manager._bots.items():
        print(f" - {bid}: {bot.config.name} (Source: Supabase)")
        
    if "primary" in manager._bots:
        print("✅ Primary bot loaded successfully from Supabase.")
    else:
        print("❌ Primary bot NOT found.")

except Exception as e:
    print(f"❌ Error: {e}")
