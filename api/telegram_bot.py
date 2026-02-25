import os
import asyncio
import threading
import logging
import json
import time
import requests
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)


class TelegramBot:
    """Telegram bot bridge that uses `requests` for ALL outbound API calls.

    python-telegram-bot (httpx) is NOT used for any outbound traffic because
    Hugging Face Spaces intermittently blocks outbound TCP to Telegram IPs.
    We only keep the Application object for parsing incoming webhook updates,
    and even the handler replies go through `requests`.
    """

    API = "https://api.telegram.org"

    def __init__(self, token: str, bot_instance=None):
        self.token = token
        self.bot_instance = bot_instance
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.chat_id: Optional[int] = None
        self._ready = False  # True once webhook is set
        self._load_chat_id()

    # ── helpers ──────────────────────────────────────────────────────────

    def _load_chat_id(self):
        if self.bot_instance and getattr(self.bot_instance.config, "telegram_chat_id", None):
            self.chat_id = self.bot_instance.config.telegram_chat_id
            self._log(f"Loaded chat_id from bot config: {self.chat_id}")

    def _save_chat_id(self, chat_id: int):
        self.chat_id = chat_id
        if self.bot_instance:
            self.bot_instance.config.telegram_chat_id = chat_id
            try:
                from api.live_bot import bot_manager
                bot_manager.save_bots()
            except Exception:
                pass
            self._log(f"Saved chat_id to bot config: {chat_id}")

    def _log(self, msg: str):
        print(f"[TELEGRAM] {msg}")

    def _api(self, method: str, payload: dict = None, retries: int = 3) -> dict:
        """Call any Telegram Bot API method via `requests` with retries."""
        url = f"{self.API}/bot{self.token}/{method}"
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(url, json=payload or {}, timeout=30)
                data = resp.json()
                if data.get("ok"):
                    return data
                else:
                    self._log(f"API {method} error: {data}")
                    return data
            except Exception as e:
                self._log(f"API {method} attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    time.sleep(min(attempt * 3, 15))
        return {"ok": False, "description": "All retries exhausted"}

    # ── DNS fix ──────────────────────────────────────────────────────────

    def _fix_telegram_dns(self):
        """Sync DNS fix — resolve via DoH and inject into /etc/hosts."""
        import socket
        import urllib.request

        def _resolve_via_doh(hostname):
            apis = [
                f"https://cloudflare-dns.com/dns-query?name={hostname}&type=A",
                f"https://dns.google/resolve?name={hostname}&type=A"
            ]
            for api_url in apis:
                try:
                    req = urllib.request.Request(api_url, headers={"Accept": "application/dns-json"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        data = json.loads(resp.read().decode())
                        for ans in data.get("Answer", []):
                            if ans.get("type") == 1:
                                return ans["data"]
                except Exception:
                    pass
            return None

        def _inject(hostname, ip):
            try:
                with open("/etc/hosts", "r") as f:
                    if hostname in f.read():
                        return True
                with open("/etc/hosts", "a") as f:
                    f.write(f"\n{ip} {hostname}\n")
                self._log(f"Wrote to /etc/hosts: {ip} {hostname}")
                return True
            except Exception:
                return False

        try:
            socket.gethostbyname("api.telegram.org")
            self._log("DNS working natively for api.telegram.org")
        except Exception:
            self._log("Local DNS failed — trying DoH...")
            ip = _resolve_via_doh("api.telegram.org")
            if ip:
                self._log(f"DoH resolved api.telegram.org -> {ip}")
                _inject("api.telegram.org", ip)
            else:
                self._log("WARNING: DoH also failed to resolve api.telegram.org")

    # ── outbound messaging ───────────────────────────────────────────────

    def send_notification(self, message: str):
        """Send a message to the chat. Uses `requests` — no httpx."""
        if self.bot_instance and getattr(self.bot_instance.config, "telegram_chat_id", None):
            self.chat_id = self.bot_instance.config.telegram_chat_id

        if not self.chat_id or not self.token:
            self._log("Cannot send notification: No chat_id or token.")
            return

        result = self._api("sendMessage", {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        })
        if result.get("ok"):
            self._log(f"Notification sent to {self.chat_id}")

    def _reply(self, chat_id: int, text: str):
        """Reply to a specific chat (used by command handlers)."""
        self._api("sendMessage", {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        })

    # ── webhook update handling (no outbound httpx needed) ───────────────

    async def handle_webhook_update(self, data: dict):
        """Process an incoming webhook update — pure Python, no httpx."""
        if not self._ready:
            self._log("Webhook received but bot not ready. Queuing in thread...")

        try:
            update_id = data.get("update_id", "?")
            self._log(f"Processing webhook update: {update_id}")

            message = data.get("message", {})
            text = message.get("text", "")
            chat_id = message.get("chat", {}).get("id")

            if not chat_id:
                self._log(f"Update {update_id}: no chat_id, skipping.")
                return

            # Route commands
            if text.startswith("/start") or text.startswith("/help"):
                self._handle_start(chat_id)
            elif text.startswith("/status"):
                self._handle_status(chat_id)
            elif text.startswith("/positions"):
                self._handle_positions(chat_id)
            elif text.startswith("/trades"):
                self._handle_trades(chat_id)
            else:
                self._log(f"Update {update_id}: unrecognised text, ignoring.")

            self._log(f"Successfully processed update: {update_id}")
        except Exception as e:
            self._log(f"Webhook processing error: {e}")
            import traceback
            traceback.print_exc()

    # ── command handlers (all use `requests` via _reply) ─────────────────

    def _handle_start(self, chat_id: int):
        self._save_chat_id(chat_id)
        self._reply(chat_id,
            "🚀 *Artoro Trading Bot connected!*\n\n"
            "I will send you notifications for every trade.\n\n"
            "*Commands:*\n"
            "/status - Current bot status & balance\n"
            "/positions - View open positions\n"
            "/trades - Last 5 trades\n"
            "/help - Show this message"
        )

    def _handle_status(self, chat_id: int):
        if not self.bot_instance:
            self._reply(chat_id, "Bot instance not available.")
            return

        status = self.bot_instance.get_status()
        state = status.get("status", "unknown").upper()

        balance_text = "N/A"
        try:
            account = self.bot_instance.api.get_account()
            balance_text = f"${float(account.equity):.2f} (Cash: ${float(account.cash):.2f})"
        except Exception:
            pass

        self._reply(chat_id,
            f"🤖 *Status:* {state}\n"
            f"💰 *Equity:* {balance_text}\n"
            f"🕒 *Last Scan:* {status.get('last_scan') or 'Never'}\n"
            f"📈 *Coins:* {len(status.get('config', {}).get('coins', []))}"
        )

    def _handle_positions(self, chat_id: int):
        if not self.bot_instance or not self.bot_instance.api:
            self._reply(chat_id, "Bot API not available.")
            return

        try:
            positions = self.bot_instance.api.list_positions()
            if not positions:
                self._reply(chat_id, "No open positions.")
                return

            msg = "📊 *Open Positions:*\n\n"
            for p in positions:
                symbol = p.symbol
                qty = float(p.qty)
                entry = float(p.avg_entry_price)
                current = float(p.current_price)
                pnl = float(p.unrealized_pl)
                pnl_pct = float(p.unrealized_plpc) * 100

                emoji = "🟢" if pnl > 0 else "🔴"
                msg += f"{emoji} *{symbol}*\n   Qty: {qty}\n   Entry: ${entry:.2f}\n   Now: ${current:.2f}\n   PnL: ${pnl:.2f} ({pnl_pct:.2f}%)\n\n"

            self._reply(chat_id, msg)
        except Exception as e:
            self._reply(chat_id, f"Error fetching positions: {e}")

    def _handle_trades(self, chat_id: int):
        if not self.bot_instance:
            self._reply(chat_id, "Bot instance not available.")
            return

        trades = list(self.bot_instance._trades)[-5:]
        if not trades:
            self._reply(chat_id, "No recent trades.")
            return

        msg = "📜 *Recent Trades:*\n\n"
        for t in reversed(trades):
            action = t.get("action")
            symbol = t.get("symbol")
            price = t.get("price") or 0
            pnl = t.get("pnl", 0)
            ts = t.get("timestamp", "").split("T")[0]

            icon = "🛒" if action == "BUY" else "💰"
            pnl_text = f" | PnL: ${pnl:.2f}" if action == "SELL" else ""
            msg += f"{icon} {action} {symbol} @ ${price:.2f}{pnl_text} ({ts})\n"

        self._reply(chat_id, msg)

    # ── lifecycle ────────────────────────────────────────────────────────

    def stop(self):
        self._log("Stopping Telegram Bot...")
        self._ready = False

    def run(self):
        """Background thread: fix DNS, set webhook, then sleep forever."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            self._log("Starting Telegram Bot bridge...")

            # DNS fix first
            self._fix_telegram_dns()

            # Wait for network to stabilise on HF
            self._log("Waiting 10s for network baseline...")
            time.sleep(10)

            # Set webhook via requests
            webhook_url = os.getenv("WEBHOOK_URL")
            if webhook_url:
                hook_path = f"{webhook_url.rstrip('/')}/tg-webhook/{self.token}"
                self._log(f"Setting webhook to: {hook_path}")

                max_retries = 10
                for attempt in range(1, max_retries + 1):
                    self._log(f"Telegram init attempt {attempt}/{max_retries}...")
                    result = self._api("setWebhook", {"url": hook_path})
                    if result.get("ok"):
                        self._log("SUCCESS: Webhook set!")
                        break
                    else:
                        self._log(f"Attempt {attempt} failed: {result}")
                        if attempt == max_retries:
                            self._log("WARNING: Could not set webhook after all retries.")
                        time.sleep(min(attempt * 5, 30))

                # Register commands
                self._api("setMyCommands", {"commands": [
                    {"command": "start", "description": "Start the bot"},
                    {"command": "status", "description": "Bot status & balance"},
                    {"command": "positions", "description": "Open positions"},
                    {"command": "trades", "description": "Recent trades"},
                    {"command": "help", "description": "Help"},
                ]})

                # Verify bot identity
                me = self._api("getMe")
                if me.get("ok"):
                    username = me["result"].get("username", "?")
                    self._log(f"Connection verified! Bot is @{username}")
                
                self._ready = True
                self._log("Telegram bridge is READY.")

                # Keep thread alive
                self.loop.run_forever()
            else:
                self._log("No WEBHOOK_URL set. Telegram bridge inactive.")
        except Exception as e:
            self._log(f"Fatal error in Telegram thread: {e}")
        finally:
            self._log("Telegram thread exiting.")

    def start_in_thread(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()


# Helper to start it easily
def start_telegram_bridge(token: str, bot_instance):
    bridge = TelegramBot(token, bot_instance)
    bridge.start_in_thread()
    return bridge
