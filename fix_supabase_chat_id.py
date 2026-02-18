import os
import sys
import json
from dotenv import load_dotenv

# Add the project root to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

# Load environment variables
load_dotenv(os.path.join(base_dir, ".env"))

try:
    import api.stock_ai as stock_ai
    from datetime import datetime, timezone
    
    print("Initializing Supabase...")
    stock_ai._init_supabase()
    
    if not stock_ai.supabase:
        print("Error: Supabase client not initialized. Check your .env for NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
        sys.exit(1)
        
    print("Fetching primary bot config directly from Supabase...")
    res = stock_ai.supabase.table("bot_configs").select("*").eq("bot_id", "primary").execute()
    
    if not res.data:
        print("Error: Primary bot config not found in Supabase.")
        sys.exit(1)
        
    record = res.data[0]
    config = record.get("config", {})
    
    old_chat_id = config.get("telegram_chat_id")
    new_chat_id = -1003699330518
    
    print(f"Current chat_id in Supabase: {old_chat_id}")
    print(f"Changing to: {new_chat_id}")
    
    config["telegram_chat_id"] = new_chat_id
    
    print("Updating Supabase table...")
    update_res = stock_ai.supabase.table("bot_configs").update({
        "config": config,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }).eq("bot_id", "primary").execute()
    
    if update_res.data:
        print("SUCCESS: Supabase record updated directly.")
        print(f"Confirmed new chat_id in DB: {update_res.data[0]['config']['telegram_chat_id']}")
        
        # Also update local state files
        print("Syncing to local files...")
        os.makedirs("state", exist_ok=True)
        with open("state/telegram_chat_id.json", "w") as f:
            json.dump({"chat_id": new_chat_id}, f)
        print("Local state file updated.")
        
    else:
        print("FAILED: No data returned after update.")
        
except Exception as e:
    print(f"FAILED with exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
