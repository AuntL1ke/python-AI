
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Генеруємо дані для навчання
X = np.linspace(-20, 20, 40000).reshape(-1, 1)  # багато точок між -20 та 20
Y = np.sin(X).flatten() + 0.1 * X.flatten()**2 + np.random.normal(0, 0.5, size=X.shape[0])

# 3. Створюємо модель: Поліноміальна регресія ступеня 3
degree = 3
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# 4. Навчаємо модель на всіх даних
model.fit(X, Y)

# 5. Генеруємо тестові дані в межах [-20, 20]
X_test = np.linspace(-20, 20, 10000).reshape(-1, 1)
Y_test = np.sin(X_test).flatten() + 0.1 * X_test.flatten()**2 + np.random.normal(0, 0.5, size=X_test.shape[0])

# 6. Передбачення на тестових даних
y_pred = model.predict(X_test)

# 7. Побудова графіку
plt.figure(figsize=(10, 6))
plt.plot(X_test, Y_test, label='Real function (з шумом)', color='blue')
plt.plot(X_test, y_pred, label='Predicted function (Poly degree 3)', color='red', linestyle='--')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Real vs Predicted Function (-20 to 20)')
plt.legend()
plt.grid(True)
plt.show()

# 8. Передбачення для конкретної точки
x_value = 7
predicted_value = model.predict(np.array([[x_value]]))
print(f"Predicted value at x={x_value}: {predicted_value[0]:.2f}")

# 9. Обчислюємо метрики
mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2): {r2:.4f}")
