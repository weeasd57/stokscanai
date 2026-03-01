# Mock Supabase before any imports that might trigger it
import os
import sys
from unittest.mock import MagicMock
from datetime import datetime, timezone

# Add project root to sys.path to import api modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

mock_supabase = MagicMock()
sys.modules['supabase'] = MagicMock()
# Mock api.stock_ai if it's already triggered
mock_stock_ai = MagicMock()
mock_stock_ai.supabase = mock_supabase
sys.modules['api.stock_ai'] = mock_stock_ai

from api.live_bot import LiveBot, BotConfig

def test_cornix_webhook_logic():
    print("Testing Cornix Webhook Logic...")
    
    # Mock methods that hit database/external APIs
    LiveBot._load_bot_state = MagicMock()
    LiveBot._load_persistent_data = MagicMock()
    LiveBot._build_default_config = MagicMock(return_value=BotConfig(name="TestBot"))
    
    config = BotConfig(name="TestBot")
    bot = LiveBot(config=config)
    bot._log = MagicMock()
    bot._save_bot_state = MagicMock()
    bot._save_trade_persistent = MagicMock()
    
    # 1. Test trade_opened
    print("Submitting trade_opened event...")
    event_open = {
        "action": "trade_opened",
        "symbol": "BTC/USDT",
        "price": 60000.0,
        "amount": 0.1
    }
    bot.handle_cornix_event(event_open)
    
    norm = "BTCUSDT"
    assert norm in bot._pos_state
    assert bot._pos_state[norm]["entry_price"] == 60000.0
    assert bot._pos_state[norm]["amount"] == 0.1
    print("trade_opened success!")
    
    # 2. Test target_hit
    print("Submitting target_hit event...")
    event_tp = {
        "action": "target_hit",
        "symbol": "BTC/USDT",
        "price": 62000.0,
        "pnl": 200.0
    }
    bot.handle_cornix_event(event_tp)
    
    assert norm not in bot._pos_state
    # Check if a trade was recorded (last one in _trades should be the sell)
    assert len(bot._trades) > 0
    last_trade = bot._trades[-1]
    assert last_trade["symbol"] == "BTC/USDT"
    assert last_trade["action"] == "SELL"
    assert last_trade["price"] == 62000.0
    assert last_trade["pnl"] == 200.0
    assert last_trade["metadata"]["reason"] == "TARGET_HIT"
    print("target_hit success!")

    # 3. Test trade_closed (directly)
    print("Submitting trade_opened and then trade_closed...")
    bot.handle_cornix_event(event_open)
    assert norm in bot._pos_state
    
    event_close = {
        "action": "trade_closed",
        "symbol": "BTC/USDT",
        "price": 61000.0,
        "pnl": 100.0
    }
    bot.handle_cornix_event(event_close)
    assert norm not in bot._pos_state
    last_trade = bot._trades[-1]
    assert last_trade["metadata"]["reason"] == "TRADE_CLOSED"
    print("trade_closed success!")

    # 4. Test direct signal dispatch via webhook
    print("Testing direct signal dispatch via webhook...")
    bot.config.cornix_webhook_url = "https://mock.webhook.url"
    
    # Mock httpx.Client.post
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    with MagicMock() as mock_client:
        mock_client.__enter__.return_value.post.return_value = mock_response
        sys.modules['httpx'] = MagicMock()
        import httpx
        httpx.Client = MagicMock(return_value=mock_client)
        
        # Trigger signal
        bot._send_telegram_signal("BTC/USDT", 60000.0, action="BUY")
        
        # Verify post was called
        mock_client.__enter__.return_value.post.assert_called()
        args, kwargs = mock_client.__enter__.return_value.post.call_args
        payload = kwargs['json']
        assert payload['action'] == 'buy'
        assert payload['symbol'] == 'BTC/USDT'
        assert payload['price'] == 60000.0
        print("Direct signal dispatch success!")

    print("\nAll Cornix logic tests passed!")

if __name__ == "__main__":
    try:
        test_cornix_webhook_logic()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
