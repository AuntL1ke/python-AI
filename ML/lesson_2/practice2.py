import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Завантаження CSV-файлу ===
df = pd.read_csv('orders_500.csv')

# Конвертація дати
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# === 2. Обчислення TotalAmount ===
df['TotalAmount'] = df['Quantity'] * df['Price']

# === 3a. Загальний дохід магазину ===
total_income = df['TotalAmount'].sum()
print(f"Загальний дохід магазину: {total_income}")

# === 3b. Середній чек (TotalAmount) ===
average_amount = df['TotalAmount'].mean()
print(f"Середній чек: {average_amount:.2f}")

# === 3c. Кількість замовлень по кожному клієнту ===
print("\nКількість замовлень по кожному клієнту:")
print(df['Customer'].value_counts())

# === 4. Замовлення з TotalAmount > 500 ===
print("\nЗамовлення, де TotalAmount > 500:")
print(df[df['TotalAmount'] > 500])

# === 5. Сортування за датою у зворотному порядку ===
print("\nЗамовлення у зворотному порядку за датою:")
print(df.sort_values(by='OrderDate', ascending=False))

# === 6. Замовлення з 5 по 10 червня (включно) ===
print("\nЗамовлення з 5 по 10 червня:")
mask = (df['OrderDate'] >= '2023-06-05') & (df['OrderDate'] <= '2023-06-10')
print(df[mask])

# === 7. Групування за категорією ===
category_stats = df.groupby('Category').agg(
    TotalItems=('Quantity', 'sum'),
    TotalSales=('TotalAmount', 'sum')
)
print("\nСтатистика по категоріях:")
print(category_stats)

# === 8. ТОП-3 клієнти за сумою покупок ===
top_customers = df.groupby('Customer')['TotalAmount'].sum().sort_values(ascending=False).head(3)
print("\nТОП-3 клієнти за сумою покупок:")
print(top_customers)

# === Завдання 2 ===

# 📈 Графік кількості замовлень по датах
orders_per_day = df.groupby('OrderDate').size()
plt.figure(figsize=(10, 5))
orders_per_day.plot(marker='o')
plt.title("Кількість замовлень по датах")
plt.xlabel("Дата")
plt.ylabel("Кількість замовлень")
plt.grid(True)
plt.tight_layout()
plt.show()

# 📊 Діаграма розподілу доходів по категоріях
plt.figure(figsize=(8, 5))
sns.barplot(x='Category', y='TotalAmount', data=df, estimator=sum, ci=None)
plt.title("Сумарні доходи по категоріях")
plt.ylabel("Сума продажів")
plt.xlabel("Категорія")
plt.tight_layout()
plt.show()
