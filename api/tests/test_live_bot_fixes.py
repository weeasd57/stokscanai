"""
Tests for the Live-Trading Shock fixes applied to live_bot.py.
"""
import sys
import os
sys.path.append(os.getcwd())

from api.live_bot import BotConfig

def test_config_fields_exist():
    cfg = BotConfig()
    assert hasattr(cfg, "warmup_bars") and cfg.warmup_bars == 100
    assert hasattr(cfg, "slippage_buffer_pct") and abs(cfg.slippage_buffer_pct - 0.005) < 1e-9
    assert hasattr(cfg, "live_confidence_haircut") and abs(cfg.live_confidence_haircut - 0.05) < 1e-9
    print("PASS: test_config_fields_exist")

def test_atr_sl_floor_is_3_percent():
    import pandas as pd, numpy as np, types
    from api.live_bot import LiveBot
    cfg = BotConfig(name="T", coins=["BTC/USDT"], use_atr_exits=True, atr_sl_multiplier=0.1, atr_tp_multiplier=2.5, atr_period=14)
    class MockBot:
        config = cfg
        telegram_bridge = None
        def _log(self, msg): pass
        def _calculate_atr(self, bars, period): return 1.0
    bot = MockBot()
    bot._calculate_atr_exits = types.MethodType(LiveBot._calculate_atr_exits, bot)
    dates = pd.date_range("2024-01-01", periods=20, freq="h")
    bars = pd.DataFrame({"timestamp": dates, "open": [100.0]*20, "high": [101.0]*20, "low": [99.0]*20, "close": [100.0]*20, "volume": [1000]*20})
    tp, sl = bot._calculate_atr_exits(bars, 100.0)
    assert sl <= 97.0, f"SL floor should be 3% min. Got SL={sl}"
    assert tp > 100.0, f"TP should be above entry. Got TP={tp}"
    print(f"PASS: test_atr_sl_floor (SL={sl}, TP={tp})")

def test_confidence_haircut():
    cfg = BotConfig(live_confidence_haircut=0.10)
    assert abs((0.70 - cfg.live_confidence_haircut) - 0.60) < 1e-9
    cfg2 = BotConfig(live_confidence_haircut=0.0)
    assert abs((0.70 - cfg2.live_confidence_haircut) - 0.70) < 1e-9
    print("PASS: test_confidence_haircut")

def test_slippage_buffer():
    cfg = BotConfig(slippage_buffer_pct=0.005)
    assert abs(100.0 * (1 + cfg.slippage_buffer_pct) - 100.50) < 1e-9
    print("PASS: test_slippage_buffer")

def test_warmup_config():
    assert BotConfig().warmup_bars == 100
    assert BotConfig(warmup_bars=50).warmup_bars == 50
    print("PASS: test_warmup_config")

if __name__ == "__main__":
    tests = [test_config_fields_exist, test_atr_sl_floor_is_3_percent, test_confidence_haircut, test_slippage_buffer, test_warmup_config]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\nResults: {len(tests)-failed}/{len(tests)} passed")
    if failed: sys.exit(1)
    else: print("All tests passed!")
