# dialogue_manager.py
import nltk
import os

dialogues = []

def load_dialogues(file_path):
    global dialogues
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден.")
        return
    with open(file_path, encoding='utf-8') as f:
        blocks = f.read().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 2:
                question = lines[0].strip('- ').strip()
                answer = lines[1].strip('- ').strip()
                if question and answer:
                    dialogues.append((question.lower(), answer))

def generate_dialogue_answer(replica):
    replica = replica.lower()
    answers = []
    for question, answer in dialogues:
        if len(question) == 0:
            continue
        if abs(len(replica) - len(question)) / len(question) < 0.33:
            distance = nltk.edit_distance(replica, question)
            distance_weighted = distance / len(question)
            if distance_weighted < 0.4:
                answers.append((distance_weighted, answer))
    if answers:
        return min(answers, key=lambda x: x[0])[1]
    return None