import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'api'))
from stock_ai import _init_supabase, supabase
from dotenv import load_dotenv

load_dotenv()
_init_supabase()

if supabase:
    res = supabase.table('scan_results').select('symbol, status, exit_price, profit_loss_pct, updated_at').in_('symbol', ['EITP', 'AFMC', 'APPC']).execute()
    for row in res.data:
        print(f"Symbol: {row['symbol']}, Status: {row['status']}, Exit: {row['exit_price']}, P/L: {row['profit_loss_pct']}%")
else:
    print("Failed to initialize Supabase")
