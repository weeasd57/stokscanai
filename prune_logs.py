import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")

if not URL or not KEY:
    print("Error: NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not found in .env")
    exit(1)

supabase: Client = create_client(URL, KEY)

def prune_logs(bot_id: str = "primary"):
    print(f"Pruning logs for bot: {bot_id}...")
    try:
        # Get count before
        resp = supabase.table("bot_logs").select("count", count="exact").eq("bot_id", bot_id).execute()
        count_before = resp.count if resp else 0
        print(f"Records before pruning: {count_before}")

        if count_before == 0:
            print("No logs to prune.")
            return

        # Simple deletion of all logs for this bot
        # The user wanted to remove all logs for the 'primary' bot to optimize
        res = supabase.table("bot_logs").delete().eq("bot_id", bot_id).execute()
        
        # Verify
        resp_after = supabase.table("bot_logs").select("count", count="exact").eq("bot_id", bot_id).execute()
        count_after = resp_after.count if resp_after else 0
        
        print(f"Records after pruning: {count_after}")
        print(f"Successfully removed {count_before - count_after} logs.")
        
    except Exception as e:
        print(f"Error during pruning: {e}")

if __name__ == "__main__":
    prune_logs("primary")
