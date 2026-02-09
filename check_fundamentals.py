
import os
from api.stock_ai import supabase

try:
    if not supabase:
        print("Supabase client not initialized.")
        exit(1)
        
    res = supabase.table("stock_fundamentals").select("count").eq("exchange", "EGX").execute()
    print(f"Fundamentals Status: {res}")
    
    count = res.count if hasattr(res, 'count') and res.count is not None else (len(res.data) if res.data else 0)
    print(f"Found {count} records for EGX in stock_fundamentals.")
    
except Exception as e:
    print(f"Error checking fundamentals: {e}")
