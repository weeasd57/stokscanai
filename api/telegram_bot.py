import os
import asyncio
import threading
import logging
from datetime import datetime
from typing import Optional
# import nest_asyncio
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from pathlib import Path
import json

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce noise from telegram and its underlying HTTP client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

class TelegramBot:
    def __init__(self, token: str, bot_instance=None):
        self.token = token
        self.bot_instance = bot_instance  # LiveBot instance
        self.application: Optional[Application] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.chat_id: Optional[int] = None # Will be set on /start
        self._load_chat_id()

    def _load_chat_id(self):
        """Load chat_id from a local file if exists."""
        p = Path("state/telegram_chat_id.json")
        if p.exists():
            try:
                with open(p, "r") as f:
                    self.chat_id = json.load(f).get("chat_id")
            except:
                pass

    def _save_chat_id(self, chat_id: int):
        """Save chat_id to keep it across restarts."""
        self.chat_id = chat_id
        os.makedirs("state", exist_ok=True)
        with open("state/telegram_chat_id.json", "w") as f:
            json.dump({"chat_id": chat_id}, f)

    async def handle_webhook_update(self, data: dict):
        """Processes an update received via webhook."""
        if not self.application or not getattr(self.application, "_initialized", False) or not self.application.running:
            self._log("Webhook received but application is not fully initialized. Ignoring update.")
            return
        
        try:
            self._log(f"Processing webhook update: {data.get('update_id')}")
            update = Update.de_json(data, self.application.bot)
            
            # Ensure we are in the right loop if possible, or just await
            await self.application.process_update(update)
            self._log(f"Successfully processed update: {data.get('update_id')}")
        except Exception as e:
            self._log(f"Webhook processing error: {e}")
            import traceback
            traceback.print_exc()

    async def start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handles /start command."""
        self._log(f"Handling /start from {update.effective_chat.id}")
        chat_id = update.effective_chat.id
        self._save_chat_id(chat_id)
        await update.message.reply_text(
            "🚀 *Artoro Trading Bot connected!*\n\n"
            "I will send you notifications for every trade.\n\n"
            "*Commands:*\n"
            "/status - Current bot status & balance\n"
            "/positions - View open positions\n"
            "/trades - Last 5 trades\n"
            "/help - Show this message",
            parse_mode='Markdown'
        )

    async def status_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handles /status command."""
        self._log(f"Handling /status from {update.effective_chat.id}")
        if not self.bot_instance:
            await update.message.reply_text("Bot instance not available.")
            return

        status = self.bot_instance.get_status()
        state = status.get("status", "unknown").upper()
        
        # Get balance from Alpaca
        balance_text = "N/A"
        try:
            account = self.bot_instance.api.get_account()
            balance_text = f"${float(account.equity):.2f} (Cash: ${float(account.cash):.2f})"
        except:
            pass

        msg = (
            f"🤖 *Status:* {state}\n"
            f"💰 *Equity:* {balance_text}\n"
            f"🕒 *Last Scan:* {status.get('last_scan') or 'Never'}\n"
            f"📈 *Coins:* {len(status.get('config', {}).get('coins', []))}"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def positions_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handles /positions command."""
        if not self.bot_instance or not self.bot_instance.api:
            await update.message.reply_text("Bot API not available.")
            return

        try:
            positions = self.bot_instance.api.list_positions()
            if not positions:
                await update.message.reply_text("No open positions.")
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
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"Error fetching positions: {e}")

    async def trades_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handles /trades command."""
        if not self.bot_instance:
            await update.message.reply_text("Bot instance not available.")
            return

        trades = list(self.bot_instance._trades)[-5:]
        if not trades:
            await update.message.reply_text("No recent trades.")
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
        
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in the telegram loop."""
        import telegram
        if isinstance(context.error, telegram.error.Conflict):
            self._log("CRITICAL CONFLICT: Another instance of this bot is already running. Stopping this instance to avoid interference.")
            # Move stop to a background task so we don't block the error handling
            asyncio.create_task(self.application.stop())
            asyncio.create_task(self.application.shutdown())
        else:
            self._log(f"Telegram Error: {context.error}")

    def stop(self):
        """Stop the bot polling and the background thread."""
        if self.application:
            self._log("Stopping Telegram Bot...")
            # Schedule the stop in the bot's own loop
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.application.stop(), self.loop)
                asyncio.run_coroutine_threadsafe(self.application.shutdown(), self.loop)
        
        if self.thread and self.thread.is_alive():
            # We don't necessarily need to join since it's a daemon thread,
            # but we want it to exit its loop.
            pass

    def send_notification(self, message: str):
        """Synchronous bridge to send a notification."""
        if not self.chat_id or not self.token:
            return

        async def _send():
            try:
                bot = Bot(token=self.token)
                await bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
            except Exception as e:
                self._log(f"Error sending notification: {e}")

        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(_send(), self.loop)
        else:
            # Fallback if loop is not running yet
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(_send())
                loop.close()
            except:
                pass

    async def post_init(self, application: Application):
        """Register commands with Telegram so they show up in the UI."""
        from telegram import BotCommand
        commands = [
            BotCommand("start", "Start the bot and get chat ID"),
            BotCommand("status", "Current bot status & balance"),
            BotCommand("positions", "View open positions"),
            BotCommand("trades", "Show last 5 trades"),
            BotCommand("help", "Show help message"),
        ]
        await application.bot.set_my_commands(commands)
        self._log("Bot commands registered.")

    def send_notification(self, message: str):
        """Synchronous bridge to send a notification."""
        # Fallback to config if not set locally
        target_chat_id = self.chat_id
        if not target_chat_id and self.bot_instance and self.bot_instance.config:
            target_chat_id = self.bot_instance.config.telegram_chat_id
            
        if not target_chat_id:
            self._log("Skipping notification: Missing chat_id (neither local nor in config).")
            return

        async def _send_async():
            try:
                # Reuse the application's bot if it's running
                if self.application and self.application.running:
                    await self.application.bot.send_message(chat_id=target_chat_id, text=message, parse_mode='Markdown')
                    self._log(f"Notification sent to {target_chat_id}")
                else:
                    # Fallback to standalone bot
                    from telegram import Bot
                    from telegram.request import HTTPXRequest
                    request = HTTPXRequest(connection_pool_size=1)
                    bot = Bot(token=self.token, request=request)
                    await bot.send_message(chat_id=target_chat_id, text=message, parse_mode='Markdown')
                    self._log(f"Notification sent to {target_chat_id} (fallback bot)")
            except Exception as e:
                self._log(f"Error sending notification: {e}")

        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(_send_async(), self.loop)
        else:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(_send_async())
                loop.close()
            except Exception as e:
                self._log(f"Fallback notification loop failed: {e}")

    def run(self):
        """Start the bot in a background thread."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # === NUCLEAR IPv4 FIX: Monkeypatch socket.getaddrinfo ===
            # This forces the whole thread to prefer IPv4 for all libraries
            import socket
            orig_getaddrinfo = socket.getaddrinfo
            def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
                # Force AF_INET (IPv4)
                return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
            socket.getaddrinfo = getaddrinfo_ipv4
            self._log("Socket monkeypatch applied: Forced IPv4 (AF_INET)")

            # Start the Bot bridge
            self._log("Starting Telegram Bot bridge...")
            
            webhook_url = os.getenv("WEBHOOK_URL")
            
            async def _set_hook():
                import socket
                import urllib.request
                import json as _json
                import httpx
                from telegram.request import HTTPXRequest

                # === Step 1: DNS & Network Fix ===
                self._log("Waiting 10s for network baseline...")
                await asyncio.sleep(10)

                def _resolve_via_doh(hostname):
                    apis = [
                        f"https://cloudflare-dns.com/dns-query?name={hostname}&type=A",
                        f"https://dns.google/resolve?name={hostname}&type=A"
                    ]
                    ips = set()
                    for api_url in apis:
                        try:
                            req = urllib.request.Request(api_url, headers={"Accept": "application/dns-json"})
                            with urllib.request.urlopen(req, timeout=10) as resp:
                                data = _json.loads(resp.read().decode())
                                for ans in data.get("Answer", []):
                                    if ans.get("type") == 1: ips.add(ans["data"])
                        except Exception: pass
                    return list(ips)

                def _test_ip(ip, port=443):
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(3)
                        sock.connect((ip, port))
                        sock.close()
                        return True
                    except Exception:
                        return False

                def _inject_hosts_entry(hostname, ip):
                    hosts_path = "/etc/hosts"
                    try:
                        with open(hosts_path, "r") as f:
                            if f"{ip} {hostname}" in f.read(): return True
                        with open(hosts_path, "a") as f:
                            f.write(f"\n{ip} {hostname}\n")
                        return True
                    except Exception: return False

                # Auto-select the best IP
                candidates = ["149.154.167.220", "149.154.166.110", "149.154.167.99", "91.108.4.110"]
                doh_ips = _resolve_via_doh("api.telegram.org")
                targets = list(set(candidates + (doh_ips or [])))
                
                self._log(f"Testing {len(targets)} candidate IPs for api.telegram.org...")
                best_ip = None
                for ip in targets:
                    if _test_ip(ip):
                        best_ip = ip
                        self._log(f"REACHABLE: {ip} responds on port 443")
                        break
                
                if best_ip:
                    _inject_hosts_entry("api.telegram.org", best_ip)
                    self._log(f"Using {best_ip} for api.telegram.org")
                else:
                    self._log("WARNING: No Telegram IPs reachable via TCP.")

                # === Step 2: Build Application ===
                from telegram.request import HTTPXRequest
                request = HTTPXRequest(connection_pool_size=10)
                
                self.application = Application.builder() \
                    .token(self.token) \
                    .post_init(self.post_init) \
                    .request(request) \
                    .build()
                
                # Add handlers
                self.application.add_handler(CommandHandler("start", self.start_handler))
                self.application.add_handler(CommandHandler("help", self.start_handler))
                self.application.add_handler(CommandHandler("status", self.status_handler))
                self.application.add_handler(CommandHandler("positions", self.positions_handler))
                self.application.add_handler(CommandHandler("trades", self.trades_handler))
                self.application.add_error_handler(self.error_handler)

                # === Step 3: Initialization Loop ===
                # During init, we also check bot info to verify connection
                max_retries = 15
                for attempt in range(1, max_retries + 1):
                    try:
                        self._log(f"Telegram init attempt {attempt}/{max_retries}...")
                        await self.application.initialize()
                        
                        # Verify we can actually talk to Telegram
                        me = await self.application.bot.get_me()
                        self._log(f"Connection verified! Bot is @{me.username}")
                        
                        await self.application.start()
                        
                        if webhook_url:
                            hook_path = f"{webhook_url.rstrip('/')}/tg-webhook/{self.token}"
                            self._log(f"Setting webhook to: {hook_path}")
                            await self.application.bot.set_webhook(url=hook_path)
                            self._log("SUCCESS: Telegram Bot initialized with Webhook!")
                        else:
                            self._log("Starting Polling...")
                            await self.application.updater.start_polling()
                            self._log("SUCCESS: Telegram Bot is Polling!")
                        return
                    except Exception as e:
                        self._log(f"Attempt {attempt} failed: {e}")
                        if attempt == max_retries: raise
                        await asyncio.sleep(min(attempt * 5, 30))
                
            self.loop.run_until_complete(_set_hook())
            self.loop.run_forever()
        except Exception as e:
            self._log(f"Fatal error in Telegram thread: {e}")
        finally:
            self._log("Telegram thread exiting.")

    def _log(self, msg: str):
        print(f"[TELEGRAM] {msg}")

    def start_in_thread(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

# Helper to start it easily
def start_telegram_bridge(token: str, bot_instance):
    bridge = TelegramBot(token, bot_instance)
    bridge.start_in_thread()
    return bridge
