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
                # HF DNS can sometimes be very slow to start up.
                async def _set_hook():
                    import socket
                    import os
                    
                    # Diagnostic: Check for proxy settings
                    http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
                    https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
                    if http_proxy or https_proxy:
                        self._log(f"Proxy detected: HTTP={http_proxy}, HTTPS={https_proxy}")

                    max_retries = 30 # Try for up to ~15-20 minutes if needed
                    self._log("Waiting 15s for network baseline...")
                    await asyncio.sleep(15)
                    
                    for attempt in range(1, max_retries + 1):
                        try:
                            # DNS Diagnostic
                            try:
                                t_ip = socket.gethostbyname("api.telegram.org")
                                self._log(f"DNS Check: api.telegram.org -> {t_ip}")
                            except Exception as dns_err:
                                self._log(f"DNS Check: api.telegram.org resolution FAILED: {dns_err}")
                                # Try a general one
                                try:
                                    g_ip = socket.gethostbyname("google.com")
                                    self._log(f"DNS Check: google.com -> {g_ip} (Internet seems OK, but Telegram is blocked/unreachable)")
                                except Exception:
                                    self._log("DNS Check: google.com resolution FAILED (Total Network Isolation?)")
                            
                            self._log(f"Telegram init attempt {attempt}/{max_retries}...")
                            
                            # If we survived until here, try the real init
                            if not getattr(self.application, "_initialized", False):
                                await self.application.initialize()
                            
                            await self.application.start()
                            await self.application.bot.set_webhook(url=hook_path)
                            self._log("SUCCESS: Telegram Bot is fully initialized and Webhook is set!")
                            return
                        except Exception as e:
                            self._log(f"Attempt {attempt} failed: {e}")
                            # Don't shutdown the app entirely, just stop if it started
                            try:
                                if self.application.running:
                                    await self.application.stop()
                            except:
                                pass
                            
                            if attempt == max_retries:
                                raise
                            
                            # Exponential backoff with a cap
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
