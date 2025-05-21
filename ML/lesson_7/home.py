# trip_model_with_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# 1. Завантажуємо дані
df = pd.read_csv("trip_duration.csv")  # має стовпці departure_time, trip_duration_min

# 2. Перетворення часу в хвилини + циклічне кодування
def time_to_min(t):
    h, m = map(int, t.split(":"))
    return h * 60 + m

df['dep_min'] = df['departure_time'].apply(time_to_min)
df['sin_t']  = np.sin(2 * np.pi * df['dep_min'] / 1440)
df['cos_t']  = np.cos(2 * np.pi * df['dep_min'] / 1440)

X = df[['sin_t', 'cos_t']].values
y = df['trip_duration_min'].values

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Масштабування
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 5. Побудова та навчання NN
tf.keras.backend.clear_session()
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(128, activation='relu'),

    layers.Dense(64, activation='relu'),
    Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, name='output')
])
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

es = EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(
    X_train_s, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# 6. Оцінка на тесті
y_pred_nn = model.predict(X_test_s).flatten()
mae_nn = mean_absolute_error(y_test, y_pred_nn)
r2_nn  = r2_score(y_test, y_pred_nn)
print(f"\nNeural Net → MAE: {mae_nn:.2f}, R2: {r2_nn:.3f}")

# 7. Поліноміальна регресія (степені 2 і 3)
for deg in (2, 3):
    poly = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('lr',   LinearRegression())
    ])
    poly.fit(X_train, y_train)
    y_pred_p = poly.predict(X_test)
    mae_p = mean_absolute_error(y_test, y_pred_p)
    r2_p  = r2_score(y_test, y_pred_p)
    print(f"Poly deg={deg} → MAE: {mae_p:.2f}, R2: {r2_p:.3f}")

# 8. Побудова графіків

# 8.1 Learning curves: loss та mae
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'],  label='train MSE')
plt.plot(history.history['val_loss'], label='val MSE')
plt.xlabel('Epoch'); plt.ylabel('MSE')
plt.title('Learning Curve (MSE)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['mae'],  label='train MAE')
plt.plot(history.history['val_mae'], label='val MAE')
plt.xlabel('Epoch'); plt.ylabel('MAE')
plt.title('Learning Curve (MAE)')
plt.legend()

plt.tight_layout()
plt.show()

# 8.2 Actual vs Predicted (NN)
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred_nn, alpha=0.5)
mn, mx = min(y_test.min(), y_pred_nn.min()), max(y_test.max(), y_pred_nn.max())
plt.plot([mn,mx],[mn,mx], 'k--')
plt.xlabel('Actual duration'); plt.ylabel('Predicted duration')
plt.title('Actual vs Predicted (NN)')
plt.tight_layout()
plt.show()

# 8.3 Прогноз над усім інтервалом часу
xx = np.linspace(0, 1440, 300)
xx_feats = np.vstack([
    np.sin(2*np.pi*xx/1440),
    np.cos(2*np.pi*xx/1440)
]).T
xx_s = scaler.transform(xx_feats)
yy = model.predict(xx_s).flatten()

def minutes_to_time(x, pos=None):
    h = int(x)//60; m = int(x)%60
    return f"{h:02d}:{m:02d}"

plt.figure(figsize=(8,4))
plt.scatter(df['dep_min'], df['trip_duration_min'],
            s=10, alpha=0.3, label='Data')
plt.plot(xx, yy, color='red', label='NN prediction')
plt.xlabel('Departure time'); plt.ylabel('Trip duration (min)')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(120))
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(minutes_to_time))
plt.title('Model vs Data (NN)')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

# 9. Прогнози для заданих часів
queries = ['10:30','00:00','02:40']
q_min = np.array([time_to_min(t) for t in queries])
Xq   = np.vstack([np.sin(2*np.pi*q_min/1440),
                   np.cos(2*np.pi*q_min/1440)]).T
Xq_s = scaler.transform(Xq)
preds = model.predict(Xq_s).flatten()

print("\nNN predictions:")
for t,p in zip(queries, preds):
    print(f"  {t} → {p:.1f} хв")
