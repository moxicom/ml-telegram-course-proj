# main.py
import logging
from ml_services import load_advertising_tagged, train_advertising_classifier, load_dialogues, dialogues
from telegram_bot.telegram_bot import run_bot


def main():
    logging.basicConfig(level=logging.INFO)

    # Инициализация
    load_dialogues("datasets/dialogues.txt")  # Указываем путь к файлу
    load_advertising_tagged("datasets/ads_dialogues_tagged.txt")
    print(f"Загружено диалогов: {len(dialogues)}")
    train_advertising_classifier()

    # Запуск бота
    run_bot()


if __name__ == "__main__":
    main()