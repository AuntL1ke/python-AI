import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Читання CSV-файлу
df = pd.read_csv('fuel_consumption_vs_speed.csv')
X = df[['speed_kmh']].values
y = df['fuel_consumption_l_per_100km'].values

# Ступені поліномів
degrees = [ 2, 3, 4]
models = {}
mse_scores = {}

# Побудова моделей
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Дані')

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)

    models[degree] = model
    mse_scores[degree] = mse

    # Побудова кривої
    X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)
    
    plt.plot(X_range, y_range_pred, label=f'Ступінь {degree} (MSE: {mse:.3f})')

plt.xlabel('Швидкість (км/год)')
plt.ylabel('Витрати пального (л/100км)')
plt.title('Поліноміальна регресія')
plt.legend()
plt.grid(True)
plt.show()

# Вибір найкращої моделі
best_degree = min(mse_scores, key=mse_scores.get)
best_model = models[best_degree]
best_poly = PolynomialFeatures(degree=best_degree)
best_poly.fit(X)
X_new = np.array([[35], [95], [140]])
X_new_poly = best_poly.transform(X_new)
predictions = best_model.predict(X_new_poly)

# Вивід результатів
print(f"Найкращий ступінь полінома: {best_degree} (MSE = {mse_scores[best_degree]:.3f})")
for speed, pred in zip([35, 95, 140], predictions):
    print(f"Прогнозовані витрати пального на {speed} км/год: {pred:.2f} л/100км")
