from configs import AD_MESSAGES, BOT_CONFIG

import random
import logging
import nltk
import os

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from dotenv import load_dotenv
from sklearn.pipeline import Pipeline


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

# ====== ADV TEXTS ======

advertising_texts = []
advertising_labels = []
advertising_responses = []

def load_advertising_tagged(file_path):
    global advertising_texts, advertising_labels, advertising_responses
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден.")
        return
    with open(file_path, encoding='utf-8') as f:
        blocks = f.read().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3 and lines[0].startswith('['):
                label = lines[0].strip('[]').strip()
                question = lines[1].strip('- ').strip()
                answer = lines[2].strip('- ').strip()
                if question and answer:  # Убедимся, что оба поля заполнены
                    advertising_texts.append(question)
                    advertising_labels.append(label)
                    advertising_responses.append(answer)

advertising_clf : None | Pipeline = None  # будет позже

def train_advertising_classifier():
    global advertising_clf
    if not advertising_texts:
        print("Нет рекламных данных для обучения.")
        return
    advertising_clf = Pipeline([
        ('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 3))),
        ('clf', LinearSVC())
    ])
    advertising_clf.fit(advertising_texts, advertising_labels)


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
        if abs(len(replica) - len(question)) / len(question) < 0.33:
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


def maybe_add_advertisement(response: str, user_text: str) -> str:
    if random.random() < 0.2:  # 20% шанс
        # Примеры ключевых слов для перехода
        food_triggers = ["голод", "еда", "вкусно", "обед", "ужин"]
        weather_triggers = ["погода", "холодно", "жарко"]

        user_text_lower = user_text.lower()
        if any(trigger in user_text_lower for trigger in food_triggers):
            ad = "Кстати, раз уж речь зашла о еде, у нас в ресторане подают отличный крем-суп из тыквы!"
        elif any(trigger in user_text_lower for trigger in weather_triggers):
            ad = "В такую погоду идеально зайти в наш ресторан и согреться горячим шоколадом с круассаном!"
        else:
            ad = random.choice(AD_MESSAGES)  # Фallback на случайный текст

        logging.getLogger().info("advertisement added")
        return f"{response}\n\n{ad}"
    return response

def maybe_reply_with_advertising_hint(text: str):
    if advertising_clf is None:
        return None
    predicted = advertising_clf.predict([text.lower()])[0]
    if predicted == "advertising_hint":
        # Найдём похожую фразу из обучающих и вернём связанный ответ
        for i, example in enumerate(advertising_texts):
            distance = nltk.edit_distance(text.lower(), example.lower())
            logging.getLogger().info(f"adv hint search for {predicted}, distance: {distance}")
            if distance / len(example) < 0.3:  # Порог схожести
                return advertising_responses[i]  # Возвращаем соответствующий ответ
    return None


def get_bot_reply(text: str):
    intent = classify_intent_with_confidence(text)
    if intent:
        logging.getLogger().info("chosed intent")
        return maybe_add_advertisement(random.choice(BOT_CONFIG['intents'][intent]['responses']), text)

    ad_hint = maybe_reply_with_advertising_hint(text)
    if ad_hint:
        logging.getLogger().info("ad_hint added")
        return ad_hint

    answer = generate_answer(text)
    if answer:
        logging.getLogger().info("answer generated")
        return maybe_add_advertisement(answer, text)

    logging.getLogger().info("failed to intent any answer. failure phrase will be sent")
    return random.choice(BOT_CONFIG['failure_phrases'])

# ====== TELEGRAM BOT ======
logging.basicConfig(level=logging.INFO)

load_dotenv()
TOKEN = os.getenv('BOT_API')  # ← вставь свой токен

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я умею болтать и немного рекламировать товары :)")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # Сохраняем историю в context.user_data
    if 'history' not in context.user_data:
        context.user_data['history'] = []
    context.user_data['history'].append(user_text)
    context.user_data['history'] = context.user_data['history'][-2:]  # Храним последние 2 сообщения
    reply = get_bot_reply(user_text)
    await update.message.reply_text(reply)

def main():
    load_dialogues("dialogues.txt")  # ← сюда положи свой файл
    print(f"Загружено диалогов: {len(dialogues)}")

    load_advertising_tagged("ads_dialogues_tagged.txt")
    train_advertising_classifier()


    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    app.run_polling()

if __name__ == '__main__':
    main()
