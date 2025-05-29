# model.py
import torch
from torch import nn
import unicodedata, string, os, glob

# === CHARACTERS ===
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# === DATA PREP ===
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

def letterToIndex(letter):
    return all_letters.find(letter)

def lineToTensor(line):
    indices = [letterToIndex(letter) for letter in unicodeToAscii(line)]
    return torch.tensor(indices, dtype=torch.long).view(-1, 1)

def findFiles(path): return glob.glob(path)

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    category_lines[category] = readLines(filename)
n_categories = len(all_categories)

# === MODEL ===
class NameClassifier(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, output_size):
        super(NameClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        out = self.fc(hidden[-1])
        return out  # raw logits (for CrossEntropyLoss)

def load_model(path='name_classify_model.pt'):
    model = NameClassifier(n_letters, 64, 128, n_categories)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict(name, model):
    with torch.no_grad():
        tensor = lineToTensor(name)
        output = model(tensor)
        top_n, top_i = output.topk(1)
        return all_categories[top_i[0].item()]
