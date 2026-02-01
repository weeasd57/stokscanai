import os
import sys
from dotenv import load_dotenv

# Explicitly load .env from the root directory and web directory
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env"))
load_dotenv(os.path.join(base_dir, "web", ".env.local"), override=True)

from api.stock_ai import run_pipeline

def test_council():
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        print("Error: EODHD_API_KEY not found")
        return

    # Try a local symbol that likely has data in Supabase/Local Cache
    ticker = "COMI.EGX"
    
    print(f"Testing THE COUNCIL for {ticker} (force_local=True)...")
    try:
        result = run_pipeline(
            api_key=api_key or "DUMMY",
            ticker=ticker,
            model_name="THE_COUNCIL",
            buy_threshold=0.6,
            force_local=True
        )
        
        print("\n--- COUNCIL RESULT ---")
        print(f"Ticker: {result['ticker']}")
        print(f"Tomorrow Prediction: {result['tomorrowPrediction']}")
        print(f"Confidence (Precision Proxy): {result['precision']}")
        print(f"Last Close: {result['lastClose']}")
        print(f"Top Reasons: {result['topReasons']}")
        print(f"Consensus: {result.get('consensus')}")
        
        # Check if tomorrowPrediction logic handled council voting correctly
        if result['tomorrowPrediction'] == 1:
            print("🚀 ACTION: STRONG BUY / BUY")
        else:
            print("✋ ACTION: WAIT / SELL")
            
    except Exception as e:
        print(f"Error during council test: {e}")
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    test_council()
