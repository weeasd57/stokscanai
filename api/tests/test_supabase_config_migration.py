import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_botmanager_migrates_legacy_config_from_supabase():
    # Fake Supabase response with legacy Alpaca keys stored in config.
    fake_res = SimpleNamespace(
        data=[
            {
                "bot_id": "legacy_bot",
                "config": {
                    "name": "Legacy Bot",
                    "execution_mode": "ALPACA",
                    "alpaca_key_id": "AKIA...",
                    "alpaca_secret_key": "SECRET",
                    "alpaca_base_url": "https://paper-api.alpaca.markets",
                    "coins": ["BTC/USD", "ETH/USD"],
                    "max_open_positions": 4,
                },
            }
        ]
    )

    with patch("api.live_bot._supabase_read_with_retry", return_value=fake_res), patch(
        "api.live_bot._supabase_upsert_with_retry"
    ):
        from api.live_bot import BotManager

        mgr = BotManager()
        bot = mgr.get_bot("legacy_bot")
        assert bot is not None

        # Legacy "ALPACA" should migrate to "VIRTUAL"
        assert bot.config.execution_mode == "VIRTUAL"
        # Ensure removed fields don't exist on config
        assert not hasattr(bot.config, "alpaca_key_id")
        assert not hasattr(bot.config, "alpaca_secret_key")
        assert not hasattr(bot.config, "alpaca_base_url")
