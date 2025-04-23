# ./app/bot.py
import random
import nltk
import pickle
import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
from data.config import CONFIG
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clear_phrase, is_meaningful_text, extract_dish_name, extract_dish_category, extract_price, Stats, logger

# Загрузка токена
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Загрузка модели для намерений
try:
    with open('models/intent_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('models/intent_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError as e:
    logger.error(f"Не найдены файлы модели для намерений: {e}")
    raise

# Загрузка модели для dialogues.txt
try:
    with open('models/dialogues_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('models/dialogues_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open('models/dialogues_answers.pkl', 'rb') as f:
        answers = pickle.load(f)
except FileNotFoundError as e:
    logger.error(f"Не найдены файлы модели для dialogues.txt: {e}")
    raise

# Классификация намерения
def classify_intent(replica):
    replica = clear_phrase(replica)
    if not replica:
        return None
    vectorized = vectorizer.transform([replica])
    intent = clf.predict(vectorized)[0]
    best_score = 0
    best_intent = None
    for intent_key, data in CONFIG['intents'].items():
        for example in data.get('examples', []):
            example = clear_phrase(example)
            if not example:
                continue
            distance = nltk.edit_distance(replica, example)
            score = 1 - distance / max(len(example), 1)
            if score > best_score and score >= 0.65:
                best_score = score
                best_intent = intent_key
    logger.info(f"Classify intent: replica='{replica}', predicted='{intent}', best_intent='{best_intent}', score={best_score}")
    return best_intent or intent if best_score >= 0.65 else None

# Получение ответа
def get_answer_by_intent(intent, replica, context):
    dish_name = context.user_data.get('current_dish')
    last_response = context.user_data.get('last_bot_response', '')
    last_intent = context.user_data.get('last_intent', '')
    dish_category = extract_dish_category(replica)
    price = extract_price(replica)

    if intent in CONFIG['intents']:
        responses = CONFIG['intents'][intent]['responses']
        if not responses:
            return None
        answer = random.choice(responses)

        if intent in ['dish_price', 'dish_availability', 'dish_info', 'order_dish']:
            if not dish_name:
                if last_response and 'Кстати, у нас есть' in last_response:
                    dish_name = extract_dish_name(last_response)
                    context.user_data['current_dish'] = dish_name
                elif dish_category:
                    suitable_dishes = [dish for dish, data in CONFIG['dishes'].items() if dish_category in data.get('categories', [])]
                    if suitable_dishes:
                        dish_name = random.choice(suitable_dishes)
                        context.user_data['current_dish'] = dish_name
                        context.user_data['state'] = 'WAITING_FOR_INTENT'
                        return f"Из {dish_category} есть {dish_name}. Хотите узнать цену, состав или наличие?"
                elif last_intent == 'menu_types':
                    for hist in context.user_data.get('history', [])[::-1]:
                        hist_dish = extract_dish_name(hist)
                        if hist_dish:
                            dish_name = hist_dish
                            context.user_data['current_dish'] = dish_name
                            break
                        hist_category = extract_dish_category(hist)
                        if hist_category:
                            suitable_dishes = [dish for dish, data in CONFIG['dishes'].items() if hist_category in data.get('categories', [])]
                            if suitable_dishes:
                                dish_name = random.choice(suitable_dishes)
                                context.user_data['current_dish'] = dish_name
                                break
                if not dish_name:
                    context.user_data['state'] = 'WAITING_FOR_DISH'
                    return "Какое блюдо или категорию вы имеете в виду?"
            if dish_name in CONFIG['dishes']:
                answer = answer.replace('[dish_name]', dish_name)
                answer = answer.replace('[price]', str(CONFIG['dishes'][dish_name]['price']))
                answer = answer.replace('[description]', CONFIG['dishes'][dish_name].get('description', 'вкусное блюдо'))
                answer += f" Что ещё интересует?"
            else:
                return "Извините, такого блюда нет в меню."

        elif intent == 'dish_recommendation':
            suitable_dishes = list(CONFIG['dishes'].keys())
            if suitable_dishes:
                dish_name = random.choice(suitable_dishes)
                context.user_data['current_dish'] = dish_name
                answer = answer.replace('[dish_name]', dish_name)
                answer += f" Хотите узнать цену или состав {dish_name}?"
            else:
                return "Извините, в меню пока нет блюд."

        elif intent == 'menu_types':
            categories = random.sample(CONFIG['categories'], min(3, len(CONFIG['categories'])))
            dishes = random.sample(list(CONFIG['dishes'].keys()), min(2, len(CONFIG['dishes'])))
            answer = f"У нас есть {', '.join(categories)} и блюда вроде {', '.join(dishes)}. Что интересно?"
            context.user_data['current_dish'] = None

        elif intent == 'yes':
            if last_intent == 'hello':
                categories = random.sample(CONFIG['categories'], min(3, len(CONFIG['categories'])))
                answer = f"Отлично! У нас есть {', '.join(categories)}. Что хотите узнать?"
            elif last_intent in ['dish_price', 'dish_info', 'dish_availability', 'order_dish']:
                if dish_name:
                    answer = f"Цена на {dish_name} — {CONFIG['dishes'][dish_name]['price']} рублей. Что ещё интересует?"
                else:
                    answer = "Назови блюдо, чтобы я рассказал подробнее!"
            elif last_intent == 'menu_types':
                dishes = random.sample(list(CONFIG['dishes'].keys()), min(2, len(CONFIG['dishes'])))
                answer = f"У нас есть {', '.join(dishes)}. Назови одно, чтобы узнать больше!"
            elif last_intent == 'offtopic':
                answer = "Хорошо, давай продолжим! Хочешь узнать про блюда?"
            else:
                answer = "Хорошо, что интересует? Блюда, цены или что-то ещё?"

        elif intent == 'no':
            context.user_data['current_dish'] = None
            context.user_data['state'] = 'NONE'
            answer = "Хорошо, какое блюдо обсудим теперь?"

        elif intent == 'filter_dishes':
            if price:
                suitable_dishes = [dish for dish, data in CONFIG['dishes'].items() if data['price'] <= price]
                if suitable_dishes:
                    dishes_list = ', '.join(suitable_dishes)
                    answer = f"До {price} рублей есть: {dishes_list}."
                else:
                    answer = f"Извините, нет блюд до {price} рублей."
            elif dish_category:
                suitable_dishes = [dish for dish, data in CONFIG['dishes'].items() if dish_category in data.get('categories', [])]
                if suitable_dishes:
                    dishes_list = ', '.join(suitable_dishes)
                    answer = f"В категории {dish_category} есть: {dishes_list}."
                    context.user_data['current_dish'] = random.choice(suitable_dishes)
                    context.user_data['state'] = 'WAITING_FOR_INTENT'
                else:
                    answer = f"Извините, нет блюд в категории {dish_category}."
            else:
                answer = "Укажите цену или категорию для фильтрации."

        # Реклама
        if intent in ['hello', 'menu_types'] and random.random() < 0.2:
            ad_dish = random.choice([d for d in CONFIG['dishes'].keys() if d != dish_name])
            answer += f" Кстати, у нас есть {ad_dish} — отличный выбор для вкусного ужина!"

        context.user_data['last_intent'] = intent
        return answer
    return None

# Ответ из dialogues.txt с TF-IDF
def generate_answer(replica, context):
    replica = clear_phrase(replica)
    if not replica or not answers:
        return None
    if not is_meaningful_text(replica):
        return None
    replica_vector = tfidf_vectorizer.transform([replica])
    similarities = cosine_similarity(replica_vector, tfidf_matrix).flatten()
    best_idx = similarities.argmax()
    if similarities[best_idx] > 0.5:
        answer = answers[best_idx]
        logger.info(f"Found in dialogues.txt: replica='{replica}', answer='{answer}', similarity={similarities[best_idx]}")
        if random.random() < 0.3:
            ad_dish = random.choice(list(CONFIG['dishes'].keys()))
            answer += f" Кстати, у нас есть {ad_dish} — очень вкусно!"
        context.user_data['last_intent'] = 'offtopic'
        return answer
    logger.info(f"No match in dialogues.txt for replica='{replica}'")
    return None

# Заглушка
def get_failure_phrase():
    dish_name = random.choice(list(CONFIG['dishes'].keys()))
    return random.choice(CONFIG['failure_phrases']).replace('[dish_name]', dish_name)

# Основная логика
def bot(replica, context):
    stats = Stats(context)
    if 'state' not in context.user_data:
        context.user_data['state'] = 'NONE'
    if 'current_dish' not in context.user_data:
        context.user_data['current_dish'] = None
    if 'last_bot_response' not in context.user_data:
        context.user_data['last_bot_response'] = None
    if 'last_intent' not in context.user_data:
        context.user_data['last_intent'] = None
    if 'history' not in context.user_data:
        context.user_data['history'] = []

    context.user_data['history'].append(replica)
    context.user_data['history'] = context.user_data['history'][-5:]

    state = context.user_data['state']
    logger.info(f"Processing: replica='{replica}', state='{state}', last_intent='{context.user_data.get('last_intent')}'")

    # Проверка на несуразный текст
    if not is_meaningful_text(replica):
        context.user_data['state'] = 'NONE'
        context.user_data['current_dish'] = None
        answer = get_failure_phrase()
        context.user_data['last_bot_response'] = answer
        stats.add('failure', replica, answer, context)
        return answer

    # Проверка цены или категории
    price = extract_price(replica)
    dish_category = extract_dish_category(replica)
    if price or dish_category:
        intent = 'filter_dishes'
        answer = get_answer_by_intent(intent, replica, context)
        if answer:
            context.user_data['last_bot_response'] = answer
            stats.add('intent', replica, answer, context)
            return answer

    # Обработка состояния
    if state == 'WAITING_FOR_DISH':
        dish_name = extract_dish_name(replica)
        if dish_name:
            context.user_data['current_dish'] = dish_name
            context.user_data['state'] = 'WAITING_FOR_INTENT'
            answer = f"Вы имеете в виду {dish_name}? Хотите узнать цену, состав или наличие?"
            context.user_data['last_bot_response'] = answer
            stats.add('intent', replica, answer, context)
            return answer
        dish_category = extract_dish_category(replica)
        if dish_category:
            suitable_dishes = [dish for dish, data in CONFIG['dishes'].items() if dish_category in data.get('categories', [])]
            if suitable_dishes:
                dish_name = random.choice(suitable_dishes)
                context.user_data['current_dish'] = dish_name
                context.user_data['state'] = 'WAITING_FOR_INTENT'
                answer = f"Из {dish_category} есть {dish_name}. Хотите узнать цену, состав или наличие?"
                context.user_data['last_bot_response'] = answer
                stats.add('intent', replica, answer, context)
                return answer
        answer = "Пожалуйста, уточните название блюда или категорию."
        context.user_data['last_bot_response'] = answer
        stats.add('failure', replica, answer, context)
        return answer

    if state == 'WAITING_FOR_INTENT':
        intent = classify_intent(replica)
        if intent in ['dish_price', 'dish_availability', 'dish_info', 'order_dish']:
            context.user_data['state'] = 'NONE'
            answer = get_answer_by_intent(intent, replica, context)
            if answer:
                context.user_data['last_bot_response'] = answer
                stats.add('intent', replica, answer, context)
                return answer
        if intent == 'yes':
            dish_name = context.user_data.get('current_dish')
            if dish_name:
                context.user_data['state'] = 'NONE'
                answer = f"Цена на {dish_name} — {CONFIG['dishes'][dish_name]['price']} рублей. Что ещё интересует?"
                context.user_data['last_bot_response'] = answer
                stats.add('intent', replica, answer, context)
                return answer
        if intent == 'no':
            context.user_data['current_dish'] = None
            context.user_data['state'] = 'NONE'
            answer = "Хорошо, какое блюдо обсудим теперь?"
            context.user_data['last_bot_response'] = answer
            stats.add('intent', replica, answer, context)
            return answer
        dish_name = context.user_data.get('current_dish', 'блюдо')
        answer = f"Что хотите узнать про {dish_name}: цену, состав или наличие?"
        context.user_data['last_bot_response'] = answer
        stats.add('failure', replica, answer, context)
        return answer

    # Проверка блюда
    dish_name = extract_dish_name(replica)
    if dish_name:
        context.user_data['current_dish'] = dish_name
        context.user_data['state'] = 'WAITING_FOR_INTENT'
        answer = f"Вы имеете в виду {dish_name}? Хотите узнать цену, состав или наличие?"
        context.user_data['last_bot_response'] = answer
        stats.add('intent', replica, answer, context)
        return answer

    # Проверка категории
    dish_category = extract_dish_category(replica)
    if dish_category:
        suitable_dishes = [dish for dish, data in CONFIG['dishes'].items() if dish_category in data.get('categories', [])]
        if suitable_dishes:
            dish_name = random.choice(suitable_dishes)
            context.user_data['current_dish'] = dish_name
            context.user_data['state'] = 'WAITING_FOR_INTENT'
            answer = f"Из {dish_category} есть {dish_name}. Хотите узнать цену, состав или наличие?"
            context.user_data['last_bot_response'] = answer
            stats.add('intent', replica, answer, context)
            return answer
        answer = f"У нас нет блюд в категории {dish_category}. Попробуйте другую категорию!"
        context.user_data['last_bot_response'] = answer
        stats.add('failure', replica, answer, context)
        return answer

    # Классификация намерения
    intent = classify_intent(replica)
    if intent:
        answer = get_answer_by_intent(intent, replica, context)
        if answer:
            context.user_data['last_bot_response'] = answer
            stats.add('intent', replica, answer, context)
            return answer

    # dialogues.txt для отвлечённых тем
    answer = generate_answer(replica, context)
    if answer:
        context.user_data['last_bot_response'] = answer
        stats.add('generate', replica, answer, context)
        return answer

    # Заглушка как последний вариант
    answer = get_failure_phrase()
    context.user_data['last_bot_response'] = answer
    stats.add('failure', replica, answer, context)
    return answer

# Голос в текст
def voice_to_text(voice_file):
    recognizer = sr.Recognizer()
    try:
        audio = AudioSegment.from_ogg(voice_file)
        audio.export('voice.wav', format='wav')
        with sr.AudioFile('voice.wav') as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data, language='ru-RU')
    except (sr.UnknownValueError, sr.RequestError, Exception) as e:
        logger.error(f"Ошибка распознавания голоса: {e}")
        return None
    finally:
        if os.path.exists('voice.wav'):
            os.remove('voice.wav')

# Текст в голос
def text_to_voice(text):
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang='ru')
        voice_file = 'response.mp3'
        tts.save(voice_file)
        return voice_file
    except Exception as e:
        logger.error(f"Ошибка синтеза речи: {e}")
        return None

# Telegram-обработчики
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['start_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = 'hello'
    await update.message.reply_text(answer)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['help_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = 'help'
    await update.message.reply_text(answer)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        answer = "Пожалуйста, отправьте текст."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
        return
    answer = bot(user_text, context)
    await update.message.reply_text(answer)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    try:
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive('voice.ogg')
        text = voice_to_text('voice.ogg')
        if text:
            answer = bot(text, context)
            voice_response = text_to_voice(answer)
            if voice_response:
                with open(voice_response, 'rb') as audio:
                    await update.message.reply_voice(audio)
                os.remove(voice_response)
            else:
                await update.message.reply_text(answer)
        else:
            answer = "Не удалось распознать голос. Попробуйте ещё раз."
            context.user_data['last_bot_response'] = answer
            await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ошибка обработки голосового сообщения: {e}")
        answer = "Произошла ошибка. Попробуйте снова."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
    finally:
        if os.path.exists('voice.ogg'):
            os.remove('voice.ogg')

def run_bot():
    if not TOKEN:
        raise ValueError("TELEGRAM_TOKEN не найден")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info("Бот запускается...")
    app.run_polling()

if __name__ == '__main__':
    run_bot()