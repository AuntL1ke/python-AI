import torch
import random
import matplotlib.pyplot as plt
from torch import nn, optim
from model import NameClassifier, all_letters, n_letters, all_categories, category_lines, lineToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DATA SPLIT ===
def split_data(category_lines, val_ratio=0.2):
    train_data, val_data = [], []
    for category in all_categories:
        names = category_lines[category]
        random.shuffle(names)
        split = int(len(names) * (1 - val_ratio))
        train_data += [(name, category) for name in names[:split]]
        val_data += [(name, category) for name in names[split:]]
    return train_data, val_data

train_data, val_data = split_data(category_lines)
print(f" Тренувальних прикладів: {len(train_data)}, Валідаційних: {len(val_data)}")

# === MODEL ===
model = NameClassifier(n_letters, 64, 128, len(all_categories)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# === UTILS ===
def category_to_tensor(category):
    return torch.tensor([all_categories.index(category)], dtype=torch.long).to(device)

def train_step(name, category):
    model.train()
    line_tensor = lineToTensor(name).to(device)
    target = category_to_tensor(category)
    output = model(line_tensor)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for name, category in val_data:
            line_tensor = lineToTensor(name).to(device)
            output = model(line_tensor)
            pred = output.argmax(dim=1).item()
            if all_categories[pred] == category:
                correct += 1
            total += 1
    return correct / total

# === TRAINING ===
n_iters = 100000
print_every = 5000
all_losses = []
best_acc = 0

for iter in range(1, n_iters + 1):
    name, category = random.choice(train_data)
    loss = train_step(name, category)

    if iter % print_every == 0:
        acc = evaluate()
        all_losses.append(loss)
        print(f"{iter}: loss={loss:.4f} | val acc={acc*100:.2f}%")
        if acc > best_acc:
            torch.save(model.state_dict(), "name_classify_model.pt")
            best_acc = acc
            print(f" Збережено модель з найкращою точністю: {acc*100:.2f}%")

# === PLOT ===
plt.plot(all_losses)
plt.title("Loss over time")
plt.xlabel("every 5,000 steps")
plt.ylabel("Loss")
plt.grid()
plt.show()
