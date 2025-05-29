# predict.py
from model import load_model, predict

rnn = load_model()
print("Введи ім’я для передбачення:")

while True:
    name = input("Ім'я: ")
    if name == '': break
    result = predict(name, rnn)
    print(f"→ Ім'я '{name}' найімовірніше з країни: {result}")
