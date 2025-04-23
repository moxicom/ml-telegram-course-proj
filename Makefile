env:
	python3 -m venv venv

act-env:
	sh venv/bin/activate

unzip-all:
	unzip data/dialogues.txt.zip -d data/