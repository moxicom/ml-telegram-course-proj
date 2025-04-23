# ./app/utils.py
import logging
import nltk
from rapidfuzz import process, fuzz
from data.config import CONFIG

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Очистка фразы
def clear_phrase(phrase):
    if not phrase:
        return ""
    phrase = phrase.lower()
    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    return ''.join(symbol for symbol in phrase if symbol in alphabet).strip()

# Проверка на осмысленность текста
def is_meaningful_text(text):
    text = clear_phrase(text)
    words = text.split()
    return any(len(word) > 2 and all(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for c in word) for word in words)

# Извлечение блюда
def extract_dish_name(replica):
    replica = clear_phrase(replica)
    if not replica:
        return None
    for dish, data in CONFIG['dishes'].items():
        if dish.lower() in replica or any(syn.lower() in replica for syn in data.get('synonyms', [])):
            return dish
        candidates = [dish] + data.get('synonyms', [])
        best_match = process.extractOne(replica, candidates, scorer=fuzz.partial_ratio)
        if best_match and best_match[1] > 85:
            return dish
    return None

# Извлечение категории
def extract_dish_category(replica):
    replica = clear_phrase(replica)
    if not replica:
        return None
    for category in CONFIG['categories']:
        category_variants = [category, category + 'ы', category[:-1] + 'ая', category[:-1] + 'и']
        for variant in category_variants:
            if variant.lower() in replica:
                return category
    return None

# Извлечение цены
def extract_price(replica):
    replica = clear_phrase(replica)
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