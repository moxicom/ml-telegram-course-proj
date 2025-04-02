# ad_manager.py
import random
import logging
import nltk
from configs import AD_MESSAGES
from .ml_training import get_advertising_clf, advertising_texts, advertising_responses


def maybe_add_advertisement(response: str, user_text: str) -> str:
    if random.random() < 1:  # 20% шанс
        food_triggers = ["голод", "еда", "вкусно", "обед", "ужин"]
        weather_triggers = ["погода", "холодно", "жарко"]

        user_text_lower = user_text.lower()
        if any(trigger in user_text_lower for trigger in food_triggers):
            ad = "Кстати, раз уж речь зашла о еде, у нас в ресторане подают отличный крем-суп из тыквы!"
        elif any(trigger in user_text_lower for trigger in weather_triggers):
            ad = "В такую погоду идеально зайти в наш ресторан и согреться горячим шоколадом с круассаном!"
        else:
            ad = random.choice(AD_MESSAGES)

        logging.getLogger().info("advertisement added")
        return f"{response}. {ad}"
    return response


def maybe_reply_with_advertising_hint(text: str):
    if get_advertising_clf() is None:
        logging.getLogger().error("No advertising classifier found")
        raise Exception("no advertising classifier found")
    predicted = get_advertising_clf().predict([text.lower()])[0]
    if predicted in ["advertising_hint", "neutral"]:
        for i, example in enumerate(advertising_texts):
            logging.getLogger("searching for a advertising_hint, neutral")
            distance = nltk.edit_distance(text.lower(), example.lower())
            logging.getLogger().info(f"adv hint search for {predicted}, distance: {distance}")
            if distance / len(example) < 0.3:
                return advertising_responses[i]
    return None