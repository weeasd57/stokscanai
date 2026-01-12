#!/usr/bin/env python3
"""
Upload local data_cache files to Supabase tables.
This script migrates:
- fund/*.json → stock_fundamentals table
- prices/*.csv → stock_prices table

Run with: python api/migrate_to_supabase.py
"""

import os
import json
import glob
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Use service role for direct DB access
CACHE_DIR = os.getenv("CACHE_DIR", "api/data_cache")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def get_exchange_dirs():
    """Get all exchange directories in data_cache"""
    if not os.path.exists(CACHE_DIR):
        return []
    return [d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))]


def upload_fundamentals(exchange: str, dry_run: bool = False):
    """Upload fund/*.json files to stock_fundamentals table"""
    fund_dir = os.path.join(CACHE_DIR, exchange, "fund")
    if not os.path.exists(fund_dir):
        print(f"  No fund directory for {exchange}")
        return 0
    
    json_files = glob.glob(os.path.join(fund_dir, "*.json"))
    uploaded = 0
    skipped = 0
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Skip error status or empty data
            if data.get("status") == "error" or not data:
                skipped += 1
                continue
            
            symbol = os.path.splitext(os.path.basename(json_path))[0]
            
            # Prepare row for upsert
            row = {
                "symbol": symbol.split(".")[0] if "." in symbol else symbol,
                "exchange": exchange,
                "name": data.get("name") or data.get("General", {}).get("Name"),
                "sector": data.get("sector") or data.get("General", {}).get("Sector"),
                "industry": data.get("industry") or data.get("General", {}).get("Industry"),
                "market_cap": data.get("market_cap") or data.get("Highlights", {}).get("MarketCapitalization"),
                "pe_ratio": data.get("pe_ratio") or data.get("Highlights", {}).get("PERatio"),
                "eps": data.get("eps") or data.get("Highlights", {}).get("EarningsShare"),
                "dividend_yield": data.get("dividend_yield") or data.get("Highlights", {}).get("DividendYield"),
                "raw_data": json.dumps(data),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Remove None values
            row = {k: v for k, v in row.items() if v is not None}
            
            if dry_run:
                print(f"    Would upload: {symbol}")
            else:
                supabase.table("stock_fundamentals").upsert(
                    row,
                    on_conflict="symbol,exchange"
                ).execute()
            
            uploaded += 1
            
        except Exception as e:
            print(f"    Error processing {json_path}: {e}")
            skipped += 1
    
    print(f"  Fundamentals: {uploaded} uploaded, {skipped} skipped")
    return uploaded


def upload_prices(exchange: str, dry_run: bool = False):
    """Upload prices/*.csv files to stock_prices table"""
    prices_dir = os.path.join(CACHE_DIR, exchange, "prices")
    if not os.path.exists(prices_dir):
        print(f"  No prices directory for {exchange}")
        return 0
    
    csv_files = glob.glob(os.path.join(prices_dir, "*.csv"))
    uploaded = 0
    total_rows = 0
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty or 'Date' not in df.columns:
                continue
            
            # Clean up: Remove TradingView limit messages
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            
            symbol_full = os.path.splitext(os.path.basename(csv_path))[0]
            symbol = symbol_full.split(".")[0] if "." in symbol_full else symbol_full
            
            # Prepare rows for batch insert
            rows = []
            for _, row in df.iterrows():
                try:
                    rows.append({
                        "symbol": symbol,
                        "exchange": exchange,
                        "date": row['Date'].strftime('%Y-%m-%d'),
                        "open": float(row['Open']) if pd.notna(row.get('Open')) else None,
                        "high": float(row['High']) if pd.notna(row.get('High')) else None,
                        "low": float(row['Low']) if pd.notna(row.get('Low')) else None,
                        "close": float(row['Close']) if pd.notna(row.get('Close')) else None,
                        "adjusted_close": float(row.get('Adjusted_close', row.get('Close'))) if pd.notna(row.get('Adjusted_close', row.get('Close'))) else None,
                        "volume": int(row['Volume']) if pd.notna(row.get('Volume')) else None,
                    })
                except Exception:
                    continue
            
            if rows:
                if dry_run:
                    print(f"    Would upload: {symbol} ({len(rows)} rows)")
                else:
                    # Batch upsert in chunks of 500
                    chunk_size = 500
                    for i in range(0, len(rows), chunk_size):
                        chunk = rows[i:i + chunk_size]
                        supabase.table("stock_prices").upsert(
                            chunk,
                            on_conflict="symbol,exchange,date"
                        ).execute()
                
                uploaded += 1
                total_rows += len(rows)
                
        except Exception as e:
            print(f"    Error processing {csv_path}: {e}")
    
    print(f"  Prices: {uploaded} files, {total_rows} rows uploaded")
    return uploaded


def clean_csv_files():
    """Remove 'Data is limited...' lines from all CSV files"""
    for exchange in get_exchange_dirs():
        prices_dir = os.path.join(CACHE_DIR, exchange, "prices")
        if not os.path.exists(prices_dir):
            continue
        
        csv_files = glob.glob(os.path.join(prices_dir, "*.csv"))
        cleaned = 0
        
        for csv_path in csv_files:
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Filter out lines containing the TradingView free tier message
                filtered = [l for l in lines if 'Data is limited by one year' not in l and l.strip()]
                
                if len(filtered) < len(lines):
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        f.writelines(filtered)
                    cleaned += 1
            except Exception:
                continue
        
        if cleaned > 0:
            print(f"Cleaned {cleaned} CSV files in {exchange}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upload local data to Supabase")
    parser.add_argument("--exchange", type=str, help="Specific exchange to upload (e.g., EGX)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without actually uploading")
    parser.add_argument("--clean-csv", action="store_true", help="Clean CSV files (remove TradingView limit messages)")
    parser.add_argument("--prices-only", action="store_true", help="Only upload prices")
    parser.add_argument("--fundamentals-only", action="store_true", help="Only upload fundamentals")
    args = parser.parse_args()
    
    if args.clean_csv:
        print("Cleaning CSV files...")
        clean_csv_files()
        return
    
    exchanges = [args.exchange] if args.exchange else get_exchange_dirs()
    
    if not exchanges:
        print(f"No exchange directories found in {CACHE_DIR}")
        return
    
    print(f"Found {len(exchanges)} exchanges: {', '.join(exchanges)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("-" * 50)
    
    total_funds = 0
    total_prices = 0
    
    for exchange in exchanges:
        print(f"\nProcessing {exchange}...")
        
        if not args.prices_only:
            total_funds += upload_fundamentals(exchange, dry_run=args.dry_run)
        
        if not args.fundamentals_only:
            total_prices += upload_prices(exchange, dry_run=args.dry_run)
    
    print("-" * 50)
    print(f"Total: {total_funds} fundamentals, {total_prices} price files uploaded")


if __name__ == "__main__":
    main()
