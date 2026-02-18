from dataclasses import dataclass, asdict
from typing import Optional, List, Any, Dict

@dataclass
class BotConfig:
    name: str = "Primary Bot"
    telegram_chat_id: Optional[int] = None
    telegram_token: Optional[str] = None
    coins: list[str] = None

def update_config_mock(config, updates: Dict[str, Any]):
    current = asdict(config)
    for k, v in updates.items():
        if k in current:
            if k == "telegram_chat_id":
                if v is not None:
                    current[k] = int(float(v))
            elif k == "telegram_token":
                if v:
                    current[k] = str(v).strip()
            else:
                current[k] = v
    return BotConfig(**current)

# Test case 1: UI sends null for chat_id, we should KEEP the old one
cfg = BotConfig(telegram_chat_id=12345, telegram_token="my_token")
print(f"Initial: {cfg}")

updates = {"telegram_chat_id": None, "telegram_token": None, "name": "Updated Name"}
new_cfg = update_config_mock(cfg, updates)
print(f"After Null Update: {new_cfg}")

assert new_cfg.telegram_chat_id == 12345
assert new_cfg.telegram_token == "my_token"
assert new_cfg.name == "Updated Name"

# Test case 2: UI sends a new token, we should UPDATE it
updates = {"telegram_token": "new_token"}
new_cfg = update_config_mock(new_cfg, updates)
print(f"After Token Update: {new_cfg}")
assert new_cfg.telegram_token == "new_token"

print("VERIFICATION SUCCESSFUL")
