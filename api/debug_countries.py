import os
import sys
import json
import io

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path (so 'import api...' works)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def debug_countries():
    print("--- Debugging Countries ---")
    
    # 1. Test local list_countries
    print("\n[1] Testing list_countries()...")
    try:
        from api.symbols_local import list_countries, _default_symbols_dir, _project_root
        print(f"Project Root: {_project_root()}")
        print(f"Symbols Dir: {_default_symbols_dir()}")
        countries = list_countries()
        print(f"Result: {len(countries)} countries found.")
        if countries:
            print(f"First 5: {countries[:5]}")
    except Exception as e:
        print(f"[ERROR] list_countries(): {e}")
        import traceback
        traceback.print_exc()

    # 2. Test get_supabase_countries
    print("\n[2] Testing get_supabase_countries()...")
    try:
        from api.stock_ai import get_supabase_countries, _init_supabase
        _init_supabase()
        from api.stock_ai import supabase
        print(f"Supabase initialized: {supabase is not None}")
        sb_countries = get_supabase_countries()
        print(f"Result: {len(sb_countries)} countries found.")
    except Exception as e:
        print(f"[ERROR] get_supabase_countries(): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_countries()
