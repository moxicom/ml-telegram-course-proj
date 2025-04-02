# bot_logic/bot_logic.py
import random
import logging
import nltk
from configs import BOT_CONFIG
from ml_services import (maybe_add_advertisement, maybe_reply_with_advertising_hint, generate_dialogue_answer,
                         train_intent_classifier)

clf, vectorizer = train_intent_classifier()


def classify_intent_with_confidence(text: str):
    text_clean = text.lower()
    intent_pred = clf.predict(vectorizer.transform([text_clean]))[0]
    for example in BOT_CONFIG['intents'][intent_pred]['examples']:
        example_clean = example.lower()
        distance = nltk.edit_distance(text_clean, example_clean)
        if distance / len(example_clean) < 0.4:
            return intent_pred
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

    answer = generate_dialogue_answer(text)
    if answer:
        logging.getLogger().info("answer generated")
        return maybe_add_advertisement(answer, text)

    logging.getLogger().info("failed to intent any answer. failure phrase will be sent")
    return random.choice(BOT_CONFIG['failure_phrases'])