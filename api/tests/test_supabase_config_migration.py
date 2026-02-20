import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_botmanager_migrates_legacy_config_from_supabase():
    # Fake Supabase response with legacy Virtual keys stored in config.
    fake_res = SimpleNamespace(
        data=[
            {
                "bot_id": "legacy_bot",
                "config": {
                    "name": "Legacy Bot",
                    "execution_mode": "Virtual",
                    "Virtual_key_id": "AKIA...",
                    "Virtual_secret_key": "SECRET",
                    "Virtual_base_url": "https://paper-api.Virtual.markets",
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

        # Legacy "Virtual" should migrate to "VIRTUAL"
        assert bot.config.execution_mode == "VIRTUAL"
        # Ensure removed fields don't exist on config
        assert not hasattr(bot.config, "Virtual_key_id")
        assert not hasattr(bot.config, "Virtual_secret_key")
        assert not hasattr(bot.config, "Virtual_base_url")
