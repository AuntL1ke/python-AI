from gensim.models import KeyedVectors

# 1. Завантаження готової моделі
model = KeyedVectors.load("models/word2vec-clean.kv")

# 2. Найбільш схожі слова до "dog"
print("\nСхожі слова до 'dog':")
similar = model.most_similar("dog", topn=10)
for word, score in similar:
    print(f"{word}: {score:.3f}")

# 3. Семантична арифметика: cat - male + female
print("\nСемантична арифметика: cat - male + female")
result1 = model.most_similar(positive=["cat", "female"], negative=["male"], topn=5)
for word, score in result1:
    print(f"{word}: {score:.3f}")

# 4. Семантична арифметика: dog - male + female
print("\nСемантична арифметика: dog - male + female")
result2 = model.most_similar(positive=["dog", "female"], negative=["male"], topn=5)
for word, score in result2:
    print(f"{word}: {score:.3f}")

# 5. Семантична арифметика: king - male + female
print("\nСемантична арифметика: king - male + female")
result2 = model.most_similar(positive=["king", "female"], negative=["male"], topn=5)
for word, score in result2:
    print(f"{word}: {score:.3f}")
