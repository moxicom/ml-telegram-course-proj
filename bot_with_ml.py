import random
import logging
import nltk
import os

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from dotenv import load_dotenv

# ====== BOT CONFIG ======
BOT_CONFIG = {
    'intents': {
        'hello': {
            'examples': ['привет', 'здравствуй', 'добрый день'],
            'responses': ['Привет! Чем могу помочь?', 'Здравствуйте!']
        },
        'bye': {
            'examples': ['пока', 'до свидания', 'увидимся'],
            'responses': ['До скорого!', 'Ещё увидимся!']
        },
        'product_info': {
            'examples': ['что у тебя есть?', 'покажи товары', 'расскажи про продукты'],
            'responses': ['У нас есть смартфоны, ноутбуки и наушники. Что интересует?']
        }
    },
    'failure_phrases': [
        'Я пока не понял... Попробуй иначе.',
        'Можешь переформулировать?',
        'Я ещё учусь, скажи проще :)'
    ]
}

# ====== ML TRAINING ======
X_text = []
y = []

for intent, data in BOT_CONFIG['intents'].items():
    for example in data['examples']:
        X_text.append(example)
        y.append(intent)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
X = vectorizer.fit_transform(X_text)

clf = LinearSVC()
clf.fit(X, y)

# ====== DIALOGUES ======
dialogues = []

def load_dialogues(file_path):
    global dialogues
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден.")
        return
    with open(file_path, encoding='utf-8') as f:
        blocks = f.read().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 2:
                question = lines[0].strip('- ').strip()
                answer = lines[1].strip('- ').strip()
                # фильтруем пустые строки
                if question and answer:
                    dialogues.append((question.lower(), answer))

def generate_answer(replica):
    replica = replica.lower()
    answers = []
    for question, answer in dialogues:
        if len(question) == 0:
            continue
        if abs(len(replica) - len(question)) / len(question) < 0.3:
            distance = nltk.edit_distance(replica, question)
            distance_weighted = distance / len(question)
            if distance_weighted < 0.4:
                answers.append((distance_weighted, answer))
    if answers:
        return min(answers, key=lambda x: x[0])[1]
    return None


# ====== RESPONSE LOGIC ======
def classify_intent_with_confidence(text: str):
    text_clean = text.lower()
    intent_pred = clf.predict(vectorizer.transform([text_clean]))[0]

    # проверим насколько реально текст похож на примеры
    for example in BOT_CONFIG['intents'][intent_pred]['examples']:
        example_clean = example.lower()
        distance = nltk.edit_distance(text_clean, example_clean)
        if distance / len(example_clean) < 0.4:
            return intent_pred

    return None  # слишком сомнительно — пусть ответит из dialogues.txt

def classify_intent(text: str):
    x = vectorizer.transform([text.lower()])
    return clf.predict(x)[0]

def get_bot_reply(text: str):
    intent = classify_intent_with_confidence(text)
    if intent:
        return random.choice(BOT_CONFIG['intents'][intent]['responses'])

    answer = generate_answer(text)
    if answer:
        return answer

    return random.choice(BOT_CONFIG['failure_phrases'])

# ====== TELEGRAM BOT ======
logging.basicConfig(level=logging.INFO)

load_dotenv()
TOKEN = os.getenv('BOT_API')  # ← вставь свой токен

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я умею болтать и немного рекламировать товары :)")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    reply = get_bot_reply(user_text)
    await update.message.reply_text(reply)

def main():
    load_dialogues("dialogues.txt")  # ← сюда положи свой файл
    print(f"Загружено диалогов: {len(dialogues)}")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    app.run_polling()

if __name__ == '__main__':
    main()
