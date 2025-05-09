import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Завантаження CSV
df = pd.read_csv('internship_candidates_cefr_final.csv')

# 2. Підготовка ознак
X = pd.get_dummies(df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']], drop_first=True)
y = df['Accepted']

# 3. Розділення на train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Логістична регресія
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# 5. Прогноз і оцінка
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Графік: EntryTestScore vs ймовірність
plt.figure(figsize=(10, 6))
plt.scatter(x_test['EntryTestScore'], y_proba, alpha=0.7, c=y_pred, cmap='coolwarm', edgecolor='k')
plt.xlabel("Entry Test Score")
plt.ylabel("Probability of Acceptance")
plt.title("Ймовірність прийняття vs Entry Test Score")
plt.colorbar(label='Predicted Class')
plt.grid(True)
plt.show()

# 7. Графік: EnglishLevel vs ймовірність
english_levels = [col for col in X.columns if col.startswith('EnglishLevel_')]
english_test_data = x_test[english_levels]
english_labels = english_test_data.idxmax(axis=1).str.replace('EnglishLevel_', '')

plt.figure(figsize=(10, 6))
plt.xticks(rotation=45)
plt.scatter(english_labels, y_proba, alpha=0.7, c=y_pred, cmap='coolwarm', edgecolor='k')
plt.xlabel("English Level")
plt.ylabel("Probability of Acceptance")
plt.title("Ймовірність прийняття vs English Level")
plt.grid(True)
plt.show()

# 8. Перевірка нового кандидата
sample_data = pd.DataFrame({
    'Experience': [10],
    'Grade': [8],
    'EnglishLevel': ['Upper-Intermediate'],
    'Age': [30],
    'EntryTestScore': [900]
})

# One-hot encoding для нового прикладу
sample_encoded = pd.get_dummies(sample_data)

# Додати відсутні колонки, щоб відповідало X.columns
for col in X.columns:
    if col not in sample_encoded.columns:
        sample_encoded[col] = 0

# Упорядкувати колонки
sample_encoded = sample_encoded[X.columns]

# Прогноз
pred = model.predict(sample_encoded)[0]
proba = model.predict_proba(sample_encoded)[0][1]

print("\n Перевірка нового кандидата:")
print("Ймовірність бути прийнятим:", round(proba * 100, 2), "%")
print("Результат:", " Прийнятий" if pred == 1 else " Не прийнятий")
