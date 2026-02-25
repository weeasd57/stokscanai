import sys
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.getcwd())

from api.live_bot import LiveBot, BotConfig

def test_stale_data_detection():
    print("Testing stale data detection...")
    
    # 1. Setup config (1h timeframe)
    cfg = BotConfig(name="TestBot", timeframe="1Hour")
    
    # 2. Create Bot instance (minimal init)
    # We use MagicMock for everything except the method under test
    bot = MagicMock(spec=LiveBot)
    bot.config = cfg
    bot._log = lambda msg: print(f"LOG: {msg}")
    
    # Manually attach the actual implementation from LiveBot
    _is_bar_stale_fn = LiveBot._is_bar_stale
    import types
    bot._is_bar_stale = types.MethodType(_is_bar_stale_fn, bot)
    
    # 3. Test with fresh data (1h ago)
    now = datetime.now(timezone.utc)
    fresh_ts = (now - timedelta(hours=1)).isoformat()
    is_stale = bot._is_bar_stale(fresh_ts)
    print(f"Fresh TS ({fresh_ts}): is_stale={is_stale}")
    assert is_stale is False, "Fresh data (1h ago) should NOT be stale for 1h timeframe"
    
    # 4. Test with stale data (5h ago)
    # Max allowed is 2.5 * 1h = 2.5h
    stale_ts = (now - timedelta(hours=5)).isoformat()
    is_stale = bot._is_bar_stale(stale_ts)
    print(f"Stale TS ({stale_ts}): is_stale={is_stale}")
    assert is_stale is True, "Data (5h ago) SHOULD be stale for 1h timeframe"
    
    # 5. Test with VERY stale data (2025)
    very_stale_ts = "2025-01-01T12:00:00Z"
    is_stale = bot._is_bar_stale(very_stale_ts)
    print(f"Very Stale TS ({very_stale_ts}): is_stale={is_stale}")
    assert is_stale is True, "2025 data must be stale"

    # 6. Test with 15m timeframe
    cfg.timeframe = "15Min"
    # Max allowed = 15m * 2.5 = 37.5m
    forty_min_ago = (now - timedelta(minutes=40)).isoformat()
    is_stale = bot._is_bar_stale(forty_min_ago)
    print(f"40m ago (15m TF): is_stale={is_stale}")
    assert is_stale is True, "40m ago should be stale for 15m timeframe"

    print("\nPASS: test_stale_data_detection")

if __name__ == "__main__":
    try:
        test_stale_data_detection()
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
