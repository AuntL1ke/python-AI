import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# === 1. Читання CSV ===
data = pd.read_csv("housing_clean_500.csv")
x = data["Area_m2"].values.reshape(-1, 1)
y = data["Price_USD"].values.reshape(-1, 1)

# === 2. Callback для MAE у $ та % ===
class PrintMetrics(Callback):
    def __init__(self, y_true):
        super().__init__()
        self.avg_price = np.mean(y_true)

    def on_epoch_end(self, epoch, logs=None):
        mae = logs['mae']
        mae_pct = (mae / self.avg_price) * 100
        print(f"Epoch {epoch + 1:04d} → Loss: {logs['loss']:.2f} | MAE: ${mae:.2f} | MAE: {mae_pct:.2f}%")

# === 3. Побудова нейромережі ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(16, activation='relu'),

    Dense(1)  # вихід без активації для регресії
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae'])

# === 4. EarlyStopping + Навчання ===
early_stop = EarlyStopping(monitor='mae', patience=100, restore_best_weights=True)

print("=== 🔧 Навчання нейронної мережі ===")
history = model.fit(x, y, epochs=2500, verbose=0, callbacks=[PrintMetrics(y), early_stop])

# === 5. Прогноз нейронки ===
y_pred_nn = model.predict(x)

# === 6. Лінійна регресія ===
lr_model = LinearRegression()
lr_model.fit(x, y)
y_pred_lr = lr_model.predict(x)

# === 7. Метрики ===
mae_nn = mean_absolute_error(y, y_pred_nn)
mse_nn = mean_squared_error(y, y_pred_nn)
mae_lr = mean_absolute_error(y, y_pred_lr)
mse_lr = mean_squared_error(y, y_pred_lr)

avg_price = np.mean(y)
mae_pct_nn = (mae_nn / avg_price) * 100
mae_pct_lr = (mae_lr / avg_price) * 100

# === 8. Вивід результатів ===
print("\n=== 📊 Оцінка моделей на тренувальних даних ===")
print(f"Нейронна мережа  → MSE: {mse_nn:,.2f}, MAE: ${mae_nn:,.2f} ({mae_pct_nn:.2f}%)")
print(f"Лінійна регресія → MSE: {mse_lr:,.2f}, MAE: ${mae_lr:,.2f} ({mae_pct_lr:.2f}%)")

# === 9. Графік прогнозу NN vs LR ===
plt.figure(figsize=(10,6))
plt.scatter(x, y, label="Реальні ціни", color="black")
plt.plot(x, y_pred_nn, label="Нейронна мережа", color="red")
plt.plot(x, y_pred_lr, label="Лінійна регресія", color="blue", linestyle="dashed")
plt.xlabel("Площа (м²)")
plt.ylabel("Ціна ($)")
plt.title("Прогноз ціни житла: нейронна мережа vs регресія")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 10. Графік зміни loss/mae по епохах ===
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Loss (MSE)', color='red')
plt.plot(history.history['mae'], label='MAE ($)', color='green')
plt.xlabel("Епоха")
plt.ylabel("Значення")
plt.title("Динаміка навчання нейронної мережі")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
