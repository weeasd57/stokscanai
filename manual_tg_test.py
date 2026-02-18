import asyncio
from telegram import Bot

async def main():
    token = "7753197178:AAGfN5SZdJTA7OjtWP_xtuCrfTyxVGXXKQA"
    chat_id = -1003699330518
    message = "🔍 Test message from Antigravity: Checking connectivity."
    
    print(f"Attempting to send message to {chat_id}...")
    try:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=message)
        print("SUCCESS: Message sent!")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(main())
