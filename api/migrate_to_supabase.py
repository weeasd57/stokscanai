"""
Script to upload master symbols to Supabase.
This is used as a backup/migration tool for the symbols_data folder.

Run with: python api/migrate_to_supabase.py
"""

import os
import json
import glob
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Use service role for direct DB access
# Relative path used for local migration only.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYMBOLS_DIR = os.path.join(BASE_DIR, "symbols_data")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def upload_master_symbols(dry_run: bool = False):
    """Upload symbols_data/*.json files to master_symbols table"""
    if not os.path.exists(SYMBOLS_DIR):
        print(f"  No symbols directory found: {SYMBOLS_DIR}")
        return 0

    # Look for files matching {Country}_all_symbols_*.json
    json_files = glob.glob(os.path.join(SYMBOLS_DIR, "*_all_symbols_*.json"))
    uploaded = 0
    total_symbols = 0

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                continue

            rows = []
            for item in data:
                # Map from EODHD/Script format to master_symbols table format
                # Example: {"Symbol": "AAPL.US", "Name": "Apple Inc", "Exchange": "US", "Country": "USA"}
                sym = item.get("Symbol") or item.get("symbol")
                if not sym: continue

                rows.append({
                    "symbol": sym,
                    "exchange": item.get("Exchange") or item.get("exchange") or "",
                    "name": item.get("Name") or item.get("name") or "",
                    "country": item.get("Country") or item.get("country") or ""
                })

            if rows:
                if dry_run:
                    print(f"    Would upload: {os.path.basename(json_path)} ({len(rows)} symbols)")
                else:
                    # Batch upsert in chunks of 1000
                    chunk_size = 1000
                    for i in range(0, len(rows), chunk_size):
                        chunk = rows[i:i + chunk_size]
                        supabase.table("master_symbols").upsert(
                            chunk,
                            on_conflict="symbol,exchange"
                        ).execute()
                
                uploaded += 1
                total_symbols += len(rows)

        except Exception as e:
            print(f"    Error processing {json_path}: {e}")

    print(f"  Master Symbols: {uploaded} files, {total_symbols} symbols uploaded")
    return total_symbols


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upload master symbols to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()
    
    print("\nProcessing Master Symbols Migration...")
    upload_master_symbols(dry_run=args.dry_run)
    print("\nMigration check complete.")


if __name__ == "__main__":
    main()
