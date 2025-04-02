# telegram_bot/telegram_bot.py
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv
import os
from bot_logic.bot_logic import get_bot_reply

load_dotenv()
TOKEN = os.getenv('BOT_API')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я умею болтать и немного рекламировать товары :)")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if 'history' not in context.user_data:
        context.user_data['history'] = []
    context.user_data['history'].append(user_text)
    context.user_data['history'] = context.user_data['history'][-2:]
    reply = get_bot_reply(user_text)
    await update.message.reply_text(reply)

def run_bot():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Бот запущен...")
    app.run_polling()