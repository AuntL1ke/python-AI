import pickle
from gensim.models import KeyedVectors

# Завантажити стару .pkl модель
with open("models/word2vec-model.pkl", "rb") as f:
    old_model = pickle.load(f)

# Зберегти напряму (це вже KeyedVectors, тому .save() працює)
old_model.save("models/word2vec-clean.kv")
