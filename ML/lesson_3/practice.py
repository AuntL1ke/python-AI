import matplotlib.pyplot as plt
import numpy as np

# Створення діапазону значень x
x = np.linspace(-10, 10, 500)

# Обчислення значень функції f(x) = x^2 * sin(x)
y = x**2 * np.sin(x)

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$f(x) = x^2 \cdot \sin(x)$', color='blue')
plt.title("Графік функції f(x) = x²·sin(x), x ∈ [-10, 10]", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)  # горизонтальна вісь
plt.axvline(0, color='black', linewidth=0.5)  # вертикальна вісь
plt.legend()
plt.show()


# Генеруємо 1000 випадкових чисел з нормальним розподілом
mean = 5     # середнє значення (μ)
std_dev = 2  # стандартне відхилення (σ)
size = 1000  # кількість елементів

data = np.random.normal(loc=mean, scale=std_dev, size=size)

# Побудова гістограми
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Гістограма випадкових чисел з нормальним розподілом (μ=5, σ=2)", fontsize=14)
plt.xlabel("Значення", fontsize=12)
plt.ylabel("Частота", fontsize=12)
plt.grid(True)
plt.show()

labels = ['Автомобіль', 'Велосипед', 'Громадський транспорт', 'Пішки']
sizes = [50, 20, 25, 5]

# Побудова кругової діаграми
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title("Відсоткове співвідношення видів транспорту", fontsize=14)
plt.axis('equal')  # кругла форма діаграми
plt.show()


# Генерація випадкових оцінок
np.random.seed(42)  # для стабільності результатів
group_A = np.random.normal(loc=70, scale=10, size=100)
group_B = np.random.normal(loc=80, scale=5, size=100)
group_C = np.random.normal(loc=65, scale=15, size=100)

# Побудова box-plot
plt.figure(figsize=(10, 6))
plt.boxplot([group_A, group_B, group_C], labels=['Група A', 'Група B', 'Група C'])
plt.title("Box-plot оцінок для трьох груп студентів", fontsize=14)
plt.ylabel("Оцінки", fontsize=12)
plt.grid(True)
plt.show()