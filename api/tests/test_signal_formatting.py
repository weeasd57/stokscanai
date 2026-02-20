
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.getcwd())

from api.live_bot import LiveBot, BotConfig

def test_precision_and_laddering():
    config = BotConfig(name="TestBot", coins=["BTC/USD"], target_pct=0.05, stop_loss_pct=0.03)
    bot = LiveBot(config)
    
    test_cases = [
        ("ELF/USDT", 0.4431),
        ("DAR/USDT", 0.1582),
        ("SHIB/USDT", 0.00002514),
        ("BTC/USDT", 52145.6)
    ]
    
    for symbol, price in test_cases:
        print(f"\n--- Testing {symbol} @ {price} ---")
        msg = bot._format_cornix_signal(symbol, price)
        print(msg)
        
        # Basic validation logic
        lines = msg.split("\n")
        entries = []
        tp = None
        sl = None
        
        in_entries = False
        in_tp = False
        in_sl = False
        
        for line in lines:
            if "Entry Targets:" in line:
                in_entries = True
                continue
            if "Take-Profit Targets:" in line:
                in_entries = False
                in_tp = True
                continue
            if "Stop Targets:" in line:
                in_tp = False
                in_sl = True
                continue
            
            if in_entries and ")" in line:
                entries.append(float(line.split(")")[1].strip()))
            if in_tp and ")" in line:
                tp = float(line.split(")")[1].strip())
            if in_sl and ")" in line:
                sl = float(line.split(")")[1].strip())
        
        # Validations
        assert len(entries) == 4, f"Expected 4 entries, got {len(entries)}"
        assert entries[0] > entries[1] > entries[2] > entries[3], f"Entries not strictly decreasing: {entries}"
        assert tp > entries[0], f"TP {tp} must be above first entry {entries[0]}"
        assert sl < entries[3], f"SL {sl} must be below last entry {entries[3]}"
        assert "Binance" in msg, "Binance missing from exchanges"
        assert "Coinbase Advanced Spot" not in msg, "Old exchange string still present"
        
        print(f"DONE: {symbol} passed validation.")

if __name__ == "__main__":
    try:
        test_precision_and_laddering()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        sys.exit(1)
