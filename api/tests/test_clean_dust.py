
import pytest
from unittest.mock import MagicMock
import os, sys

# Ensure repo root is on sys.path when running via different pytest invocations.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.live_bot import LiveBot, BotConfig

@pytest.fixture
def mock_api():
    return MagicMock()

def test_clean_dust_closes_small_positions(mock_api):
    """Test that clean_dust identifies and closes positions below threshold."""
    # Setup LiveBot with mocked API
    config = BotConfig()
    bot = LiveBot(bot_id="test_bot", config=config)
    bot.api = mock_api
    
    # Mock positions: 1 valid, 1 dust, 1 zero-value but with qty (dust)
    pos1 = MagicMock()
    pos1.symbol = "BTC/USD"
    pos1.market_value = "100.0" # Not dust
    pos1.qty = "0.01"

    pos2 = MagicMock()
    pos2.symbol = "DOGE/USD"
    pos2.market_value = "0.05" # Dust (threshold 0.10)
    pos2.qty = "1.0"

    pos3 = MagicMock()
    pos3.symbol = "SHIB/USD"
    pos3.market_value = "0" # Dust (zero value)
    pos3.qty = "100.0"
    
    mock_api.list_positions.return_value = [pos1, pos2, pos3]
    
    # Run clean_dust
    count = bot.clean_dust(threshold=0.10)
    
    # Verify results
    assert count == 2
    
    # Verify close calls
    # Should close DOGE and SHIB, but NOT BTC
    calls = mock_api.close_position.call_args_list
    closed_symbols = [c[0][0] for c in calls]
    
    assert "DOGE/USD" in closed_symbols
    assert "SHIB/USD" in closed_symbols
    assert "BTC/USD" not in closed_symbols

def test_clean_dust_handles_min_qty_error(mock_api):
    """Test that clean_dust handles 'minimal qty' errors gracefully."""
    config = BotConfig()
    bot = LiveBot(bot_id="test_bot", config=config)
    bot.api = mock_api
    
    pos1 = MagicMock()
    pos1.symbol = "TINY/USD"
    pos1.market_value = "0.0001"
    pos1.qty = "0.00000001"
    
    mock_api.list_positions.return_value = [pos1]
    
    # Simulate error on close
    mock_api.close_position.side_effect = Exception("minimal qty of order must be ...")
    
    # Run clean_dust
    count = bot.clean_dust(threshold=0.10)
    
    # Should not crash, and count should not increment (since it failed to close)
    assert count == 0
    # But it should have TRIED to close
    mock_api.close_position.assert_called_with("TINY/USD")

