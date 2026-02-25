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
        if self.bot_instance and getattr(self.bot_instance.config, "telegram_chat_id", None):
            self.chat_id = self.bot_instance.config.telegram_chat_id
            self._log(f"Loaded chat_id from bot config: {self.chat_id}")

    def _save_chat_id(self, chat_id: int):
        self.chat_id = chat_id
        if self.bot_instance:
            self.bot_instance.config.telegram_chat_id = chat_id
            from api.live_bot import bot_manager
            bot_manager.save_bots()
            self._log(f"Saved chat_id to bot config and Supabase: {chat_id}")

    async def handle_webhook_update(self, data: dict):
        """Processes an update received via webhook."""
        if not self.application or not getattr(self.application, "_initialized", False):
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
        if not self.bot_instance:
            await update.message.reply_text("Bot instance not available.")
            return

        status = self.bot_instance.get_status()
        state = status.get("status", "unknown").upper()
        
        # Get balance from Broker
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
        """Synchronous bridge to send a notification. Returns Future if loop running."""
        # Refresh chat_id from bot config if available
        if self.bot_instance and getattr(self.bot_instance.config, "telegram_chat_id", None):
            self.chat_id = self.bot_instance.config.telegram_chat_id

        if not self.chat_id or not self.token:
            self._log("Cannot send notification: No chat_id or token.")
            return None

        async def _send():
            try:
                bot = Bot(token=self.token)
                await bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
            except Exception as e:
                self._log(f"Error sending notification: {e}")
                raise e

        if self.loop and self.loop.is_running():
            return asyncio.run_coroutine_threadsafe(_send(), self.loop)
        else:
            # Fallback if loop is not running yet
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(_send())
                loop.close()
                return None # Completed synchronously
            except Exception as e:
                self._log(f"Error in sync fallback: {e}")
                raise e

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

    def run(self):
        """Start the bot in a background thread."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            self.application = Application.builder().token(self.token).post_init(self.post_init).build()
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start_handler))
            self.application.add_handler(CommandHandler("help", self.start_handler))
            self.application.add_handler(CommandHandler("status", self.status_handler))
            self.application.add_handler(CommandHandler("positions", self.positions_handler))
            self.application.add_handler(CommandHandler("trades", self.trades_handler))
            
            # Add error handler
            self.application.add_error_handler(self.error_handler)
            
            # Start the bot
            self._log("Starting Telegram Bot poll...")
            
            # Check for WEBHOOK_URL in env
            webhook_url = os.getenv("WEBHOOK_URL")
            if webhook_url:
                # Hugging Face Spaces usually forward to 7860, but let's assume we handle it in uvicorn
                hook_path = f"{webhook_url.rstrip('/')}/tg-webhook/{self.token}"
                self._log(f"Setting webhook to: {hook_path}")
                # We don't use run_webhook because we want uvicorn to handle the HTTP
                # Retry the ENTIRE init+webhook sequence.
                # HF containers often block DNS for api.telegram.org.
                # We use DNS-over-HTTPS to resolve the IP and write it to /etc/hosts.
                async def _set_hook():
                    import socket
                    import os
                    import urllib.request
                    import json as _json

                    def _resolve_via_doh(hostname):
                        """Resolve hostname via DNS-over-HTTPS (Cloudflare + Google)."""
                        apis = [
                            f"https://cloudflare-dns.com/dns-query?name={hostname}&type=A",
                            f"https://dns.google/resolve?name={hostname}&type=A"
                        ]
                        for api_url in apis:
                            try:
                                req = urllib.request.Request(api_url, headers={"Accept": "application/dns-json"})
                                with urllib.request.urlopen(req, timeout=10) as resp:
                                    data = _json.loads(resp.read().decode())
                                    answers = data.get("Answer", [])
                                    for ans in answers:
                                        if ans.get("type") == 1:  # A record
                                            return ans["data"]
                            except Exception as ex:
                                self._log(f"DoH via {api_url.split('/')[2]} failed: {ex}")
                        return None

                    def _inject_hosts_entry(hostname, ip):
                        """Write an entry to /etc/hosts so ALL resolvers see it."""
                        hosts_path = "/etc/hosts"
                        entry = f"{ip} {hostname}"
                        try:
                            # Check if already present
                            with open(hosts_path, "r") as f:
                                if hostname in f.read():
                                    self._log(f"/etc/hosts already has entry for {hostname}")
                                    return True
                            with open(hosts_path, "a") as f:
                                f.write(f"\n{entry}\n")
                            self._log(f"Wrote to /etc/hosts: {entry}")
                            return True
                        except PermissionError:
                            self._log(f"No write access to {hosts_path} — cannot inject DNS override")
                            return False
                        except Exception as ex:
                            self._log(f"Failed to write {hosts_path}: {ex}")
                            return False

                    # === Step 1: Try to fix DNS before anything else ===
                    self._log("Waiting 10s for network baseline...")
                    await asyncio.sleep(10)

                    try:
                        socket.gethostbyname("api.telegram.org")
                        self._log("DNS working natively for api.telegram.org")
                    except Exception:
                        self._log("Local DNS cannot resolve api.telegram.org — trying DoH fallback...")
                        ip = _resolve_via_doh("api.telegram.org")
                        if ip:
                            self._log(f"DoH resolved api.telegram.org -> {ip}")
                            _inject_hosts_entry("api.telegram.org", ip)
                            # Verify it worked
                            try:
                                resolved = socket.gethostbyname("api.telegram.org")
                                self._log(f"Verification: api.telegram.org now resolves to {resolved}")
                            except Exception as ve:
                                self._log(f"Verification failed even after /etc/hosts: {ve}")
                        else:
                            self._log("DoH also failed — will keep retrying with native DNS")

                    # === Step 2: Retry init + webhook ===
                    max_retries = 15
                    for attempt in range(1, max_retries + 1):
                        try:
                            self._log(f"Telegram init attempt {attempt}/{max_retries}...")
                            if not getattr(self.application, "_initialized", False):
                                await self.application.initialize()
                            await self.application.start()
                            await self.application.bot.set_webhook(url=hook_path)
                            self._log("SUCCESS: Telegram Bot is fully initialized and Webhook is set!")
                            return
                        except Exception as e:
                            self._log(f"Attempt {attempt} failed: {e}")
                            try:
                                if self.application.running:
                                    await self.application.stop()
                            except:
                                pass
                            if attempt == max_retries:
                                raise
                            wait_time = min(attempt * 10, 60)
                            self._log(f"Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                
                self.loop.run_until_complete(_set_hook())
                # Keep loop running for async tasks but don't poll
                self.loop.run_forever()
            else:
                # We use run_polling to block this thread until stopped
                self.application.run_polling(close_loop=True)
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
