
import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client

# Add api folder to path
sys.path.append(os.getcwd())

# Explicitly load .env from the root directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, ".env")
print(f"Loading env from: {env_path}")
load_dotenv(env_path)

# Also load web/.env.local if present
web_env_path = os.path.join(base_dir, "web", ".env.local")
if os.path.exists(web_env_path):
    print(f"Loading env from: {web_env_path}")
    load_dotenv(web_env_path, override=True)

url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

if not url or not key:
    print(f"Error: Supabase credentials not found in env.")
    print(f"URL: {url}, KEY: {'HIDDEN' if key else 'None'}")
    sys.exit(1)

supabase: Client = create_client(url, key)

print(f"Connecting to Supabase at {url}...")

tables_to_check = ["bot_configs", "bot_states"]

for table in tables_to_check:
    try:
        print(f"Checking table '{table}'...")
        # Try to select 1 row. If table doesn't exist, it should throw.
        res = supabase.table(table).select("*").limit(1).execute()
        print(f"✅ Table '{table}' exists.")
    except Exception as e:
        print(f"❌ Table '{table}' check failed: {e}")
        if "relation" in str(e) and "does not exist" in str(e):
             print(f"   -> You need to run the SQL for '{table}' in supabase/schema.sql")

