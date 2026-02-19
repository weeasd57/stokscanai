import os
import sys
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def test_upsert():
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("Missing Supabase credentials")
        return

    supabase: Client = create_client(url, key)
    
    # Attempt to upsert a trade with on_conflict="order_id"
    trade_record = {
        "bot_id": "test_bot",
        "timestamp": "2024-02-19T12:00:00",
        "symbol": "BTC/USD",
        "action": "BUY",
        "amount": 100.0,
        "price": 50000.0,
        "order_id": "test_order_123"
    }

    print(f"Attempting upsert into 'bot_trades' with on_conflict='order_id'...")
    try:
        res = supabase.table("bot_trades").upsert([trade_record], on_conflict="order_id").execute()
        print("Upsert SUCCESSFUL")
        print(res.data)
    except Exception as e:
        print(f"Upsert FAILED as expected: {e}")

if __name__ == "__main__":
    test_upsert()
