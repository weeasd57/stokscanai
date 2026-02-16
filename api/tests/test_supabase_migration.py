
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from dataclasses import asdict

from api.live_bot import LiveBot, BotConfig, BotManager

# Mock Supabase client
@pytest.fixture
def mock_supabase():
    with patch("api.live_bot.supabase") as mock:
        yield mock

@pytest.fixture
def mock_supabase_retry():
    with patch("api.live_bot._supabase_upsert_with_retry") as mock_upsert:
        yield mock_upsert

def test_save_config_to_supabase(mock_supabase_retry):
    """Test that _save_config_to_supabase calls upsert with correct data."""
    config = BotConfig(name="Test Bot", save_to_supabase=True)
    bot = LiveBot(bot_id="test_bot", config=config)
    
    # Trigger save
    bot._save_config_to_supabase()
    
    # Verify _supabase_upsert_with_retry was called
    assert mock_supabase_retry.called
    args, kwargs = mock_supabase_retry.call_args
    
    table_name = args[0]
    records = args[1]
    
    assert table_name == "bot_configs"
    assert len(records) == 1
    assert records[0]["bot_id"] == "test_bot"
    assert records[0]["config"]["name"] == "Test Bot"
    assert "updated_at" in records[0]

def test_save_pos_state_to_supabase(mock_supabase_retry):
    """Test that _save_pos_state_to_supabase calls upsert with correct data."""
    bot = LiveBot(bot_id="test_bot")
    bot._pos_state = {"BTC/USD": {"entry_price": 50000}}
    
    # Trigger save
    bot._save_pos_state_to_supabase()
    
    # Verify
    assert mock_supabase_retry.called
    args, kwargs = mock_supabase_retry.call_args
    
    table_name = args[0]
    records = args[1]
    
    assert table_name == "bot_states"
    assert len(records) == 1
    assert records[0]["bot_id"] == "test_bot"
    assert records[0]["state"]["BTC/USD"]["entry_price"] == 50000

def test_update_config_triggers_save(mock_supabase_retry):
    """Test that update_config triggers a Supabase save."""
    bot = LiveBot(bot_id="test_bot")
    
    updates = {"name": "New Name", "use_auto_tune": True}
    bot.update_config(updates)
    
    assert bot.config.name == "New Name"
    assert bot.config.use_auto_tune is True
    
    # Verify save was called
    assert mock_supabase_retry.called
    args, _ = mock_supabase_retry.call_args
    assert args[0] == "bot_configs"
    assert args[1][0]["config"]["name"] == "New Name"

