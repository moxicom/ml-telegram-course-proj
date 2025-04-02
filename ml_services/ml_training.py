# ml_services/ml_services.py
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from configs import BOT_CONFIG
import os

# Переменные как атрибуты модуля
advertising_texts = []
advertising_labels = []
advertising_responses = []
advertising_clf = None  # Инициализируем как None


# Обучение модели намерений
def train_intent_classifier():
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
    return clf, vectorizer


# Загрузка рекламных данных
def load_advertising_tagged(file_path):
    global advertising_texts, advertising_labels, advertising_responses
    if not os.path.exists(file_path):
        logging.getLogger().error(f"Файл {file_path} не найден.")
        raise Exception(f"file {file_path} not found.")
    with open(file_path, encoding='utf-8') as f:
        blocks = f.read().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3 and lines[0].startswith('['):
                label = lines[0].strip('[]').strip()
                question = lines[1].strip('- ').strip()
                answer = lines[2].strip('- ').strip()
                if question and answer:
                    advertising_texts.append(question)
                    advertising_labels.append(label)
                    advertising_responses.append(answer)
        logging.getLogger().info(f"advertising_texts: {len(advertising_texts)}, advertising_labels: "
                                 f"{len(advertising_labels)}, advertising_responses: {len(advertising_responses)}")


# Обучение модели рекламных намёков
def train_advertising_classifier():
    global advertising_clf
    if not advertising_texts:
        logging.getLogger().error("Нет рекламных данных для обучения.")
        raise Exception("no adds text to learn")
    advertising_clf = Pipeline([
        ('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 3))),
        ('clf', LinearSVC())
    ])
    advertising_clf.fit(advertising_texts, advertising_labels)
    logging.getLogger().info("Классификатор рекламных намёков обучен")

def get_advertising_clf() -> Pipeline | None:
    global advertising_clf
    return advertising_clf

# # Экспортируем переменные для использования в других модулях
# __all__ = ['train_intent_classifier', 'load_advertising_tagged', 'train_advertising_classifier',
#            'advertising_clf', 'advertising_texts', 'advertising_responses']