env:
	python3 -m venv venv

act-env:
	sh venv/bin/activate

unzip-all:
	unzip data/dialogues.txt.zip -d data/

retrain_docker:
	docker-compose down train_intent_model -v
	docker-compose down train_dialogues_model -v
	docker-compose up --build train_intent_model -d
	docker-compose up --build train_dialogues_model -d

retrain_locally:
	venv/bin/python3 app/train_intent_model.py
	venv/bin/python3 app/train_dialogues_model.py


rebuild_bot_docker:
	docker-compose down telegram_bot
	docker-compose up --build telegram_bot

run_locally:
	venv/bin/python3 app/bot.py