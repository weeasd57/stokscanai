
import os
from dotenv import dotenv_values

base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")
local_env_path = os.path.join(base_dir, ".env.local")

print(f"Checking {env_path}")
if os.path.exists(env_path):
    config = dotenv_values(env_path)
    print("Keys in .env:", list(config.keys()))
else:
    print(".env not found")

web_env_path = os.path.join(base_dir, "web", ".env.local")
print(f"Checking {web_env_path}")
if os.path.exists(web_env_path):
    config = dotenv_values(web_env_path)
    print("Keys in web/.env.local:", list(config.keys()))
else:
    print("web/.env.local not found")
