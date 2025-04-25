# ./app/train_intent_model.py

import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from data import config
from utils import lemmatize_phrase, logger


logger.info("Начинается обучение модели для intents")

# Подготовка данных
X_text = []
y = []
for intent, data1 in config.CONFIG['intents'].items():
    for example in data1['examples']:
        lemmatized_example = lemmatize_phrase(example)  # Лемматизируем примеры
        X_text.append(lemmatized_example)
        y.append(intent)

# Векторайзер
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), lowercase=True)
X = vectorizer.fit_transform(X_text)

# Обучение
clf = LinearSVC()
clf.fit(X, y)

# Создание директории
os.makedirs('models', exist_ok=True)

# Сохранение
with open('models/intent_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('models/intent_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

logger.info("Модель для intents обучена и сохранена в ./models/")
