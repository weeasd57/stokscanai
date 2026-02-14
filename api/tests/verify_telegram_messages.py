
import sys
import os

# Add the root directory to sys.path so we can import 'api'
sys.path.append(os.getcwd())

from unittest.mock import MagicMock
from api.live_bot import LiveBot, BotConfig

def test_message_construction():
    # Mock dependencies
    mock_bridge = MagicMock()
    
    # Create a bot instance with a basic config
    config = BotConfig(
        name="Primary Bot",
        alpaca_key_id="dummy",
        alpaca_secret_key="dummy",
        coins=["BTC/USD"]
    )
    
    bot = LiveBot(config)
    bot.telegram_bridge = mock_bridge
    
    print("Testing BUY/SELL notifications and writing to api/tests/result.txt...")
    
    with open("api/tests/result.txt", "w", encoding="utf-8") as f:
        bot.send_test_notification("buy")
        args, kwargs = mock_bridge.send_notification.call_args
        f.write("BUY Message:\n")
        f.write(args[0])
        f.write("\n\n" + "="*50 + "\n\n")
        
        bot.send_test_notification("sell")
        args, kwargs = mock_bridge.send_notification.call_args
        f.write("SELL Message:\n")
        f.write(args[0])
        f.write("\n\n" + "="*50 + "\n\n")
        
        bot.send_test_notification("signal")
        args, kwargs = mock_bridge.send_notification.call_args
        f.write("SIGNAL Message:\n")
        f.write(args[0])

if __name__ == "__main__":
    test_message_construction()
