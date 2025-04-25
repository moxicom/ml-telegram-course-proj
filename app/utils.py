# ./app/utils.py

import logging
from rapidfuzz import process, fuzz
from data.config import CONFIG
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# Очистка фразы (оставляем как есть)
def clear_phrase(phrase):
    if not phrase:
        return ""
    phrase = phrase.lower()
    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    return ''.join(symbol for symbol in phrase if symbol in alphabet).strip()

# Лемматизация и морфологический анализ
def lemmatize_phrase(phrase):
    if not phrase:
        return ""
    # Очистка текста
    cleaned_phrase = clear_phrase(phrase)
    if not cleaned_phrase:
        return ""

    # Создание объекта Natasha Doc
    doc = Doc(cleaned_phrase)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    # Лемматизация
    lemmatized_words = []
    for token in doc.tokens:
        # Лемматизируем токен
        token.lemmatize(morph_vocab)

        # Используем лемму, если она есть, иначе оригинальный текст
        lemma = token.lemma if token.lemma else token.text
        lemmatized_words.append(lemma)

    res =  ' '.join(lemmatized_words)
    # print(res)

    return res

# Проверка на осмысленность текста
def is_meaningful_text(text):
    text = clear_phrase(text)
    words = text.split()
    return any(len(word) > 2 and all(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for c in word) for word in words)

# Извлечение блюда
def extract_dish_name(replica):
    replica = lemmatize_phrase(replica)  # Используем лемматизированную фразу
    if not replica:
        return None
    for dish, data in CONFIG['dishes'].items():
        # Лемматизируем название блюда и синонимы
        dish_lemmatized = lemmatize_phrase(dish)
        synonyms_lemmatized = [lemmatize_phrase(syn) for syn in data.get('synonyms', [])]
        if dish_lemmatized in replica or any(syn in replica for syn in synonyms_lemmatized):
            return dish
        candidates = [dish] + data.get('synonyms', [])
        best_match = process.extractOne(replica, candidates, scorer=fuzz.partial_ratio)
        if best_match and best_match[1] > 85:
            return dish
    return None

# Извлечение категории
def extract_dish_category(replica):
    replica = lemmatize_phrase(replica)  # Используем лемматизированную фразу
    if not replica:
        return None
    for category in CONFIG['categories']:
        # Лемматизируем категорию
        category_lemmatized = lemmatize_phrase(category)
        category_variants = [
            category_lemmatized,
            category_lemmatized + 'ы',
            category_lemmatized[:-1] + 'ая' if category_lemmatized.endswith('а') else category_lemmatized,
            category_lemmatized[:-1] + 'и' if category_lemmatized.endswith('а') else category_lemmatized
        ]
        for variant in category_variants:
            if variant in replica:
                return category
    return None

# Извлечение цены
def extract_price(replica):
    replica = clear_phrase(replica)  # Цены не лемматизируем
    if not replica:
        return None
    words = replica.split()
    for word in words:
        if word.isdigit():
            return int(word)
    return None

# Класс для управления статистикой
class Stats:
    def __init__(self, context):
        self.context = context
        if 'stats' not in context.user_data:
            context.user_data['stats'] = {'intent': 0, 'generate': 0, 'failure': 0}
        self.stats = context.user_data['stats']

    def add(self, type, replica, answer, context):
        """Обновляет статистику, сохраняет её в context и логирует."""
        if type in self.stats:
            self.stats[type] += 1
        else:
            self.stats[type] = 1
        self.context.user_data['stats'] = self.stats
        logger.info(f"Stats: {self.stats} | Вопрос: {replica} | Ответ: {answer}")