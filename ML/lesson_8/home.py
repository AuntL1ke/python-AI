import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Завантаження або генерація даних
DATA_FILE = 'synthetic_consumption.csv'
if os.path.exists(DATA_FILE):
    df_all = pd.read_csv(DATA_FILE, parse_dates=['date'])
else:
    dfs = []
    days = np.arange(-182, 183)
    for year in range(2019, 2026):
        dates = pd.date_range(start=f"{year}-01-01", periods=365)
        consumption = 0.001 * (days ** 2) + 23
        df = pd.DataFrame({
            'date': dates,
            'consumption_kwh': consumption.round(2)
        })
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(DATA_FILE, index=False)

# 2. Фічі: доба року у синус/косинус
df_all['doy']     = df_all['date'].dt.dayofyear
df_all['sin_doy'] = np.sin(2 * np.pi * df_all['doy'] / 365)
df_all['cos_doy'] = np.cos(2 * np.pi * df_all['doy'] / 365)

X = df_all[['sin_doy', 'cos_doy']].values.astype(np.float32)
y = df_all['consumption_kwh'].values.astype(np.float32)

# 3. Розбиття та масштабування
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
sc_X = StandardScaler().fit(X_train)
X_train_s = sc_X.transform(X_train)
X_test_s  = sc_X.transform(X_test)
X_all_s   = sc_X.transform(X)

sc_y = StandardScaler().fit(y_train.reshape(-1,1))
y_train_s = sc_y.transform(y_train.reshape(-1,1))
y_test_s  = sc_y.transform(y_test.reshape(-1,1))

# 4. Визначення моделі 128-64
model = Sequential([
    Dense(256, activation='relu',
          input_shape=(X_train_s.shape[1],),
          kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(model.summary())

# 5. Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

# 6. Навчання
history = model.fit(
    X_train_s, y_train_s,
    validation_data=(X_test_s, y_test_s),
    epochs=500, batch_size=32,
    callbacks=callbacks, verbose=1
)

# 7. Прогноз і оцінка
# — для тестової частини
y_pred_test_s = model.predict(X_test_s)
y_pred_test   = sc_y.inverse_transform(y_pred_test_s).ravel()

print('Test MAE:', mean_absolute_error(y_test, y_pred_test))
print('Test MSE:', mean_squared_error(y_test, y_pred_test))

# — на всьому діапазоні (для повного графіку)
y_pred_all_s = model.predict(X_all_s)
y_pred_all   = sc_y.inverse_transform(y_pred_all_s).ravel()

# 8. Графік: Actual vs Predicted (повний діапазон)
plt.figure(figsize=(12,5))
plt.plot(df_all['date'], df_all['consumption_kwh'], label='Actual', alpha=0.3)
plt.plot(df_all['date'], y_pred_all, color='C1', lw=2, label='Predicted')
plt.title('Actual vs Predicted Energy Consumption (Full Range)')
plt.xlabel('Date')
plt.ylabel('kWh')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Графік: Actual vs Predicted (train vs test)
n_train = len(X_train_s)
plt.figure(figsize=(12,5))
plt.plot(df_all['date'][:n_train], y_pred_all[:n_train],
         color='C2', lw=2, label='Predicted (train)')
plt.plot(df_all['date'][n_train:], y_pred_all[n_train:],
         color='C3', lw=2, label='Predicted (test)')
plt.scatter(df_all['date'], df_all['consumption_kwh'],
            s=10, alpha=0.3, label='Actual')
plt.axvline(df_all['date'].iloc[n_train], color='gray',
            ls='--', alpha=0.5)
plt.text(df_all['date'].iloc[n_train],
         plt.ylim()[1]*0.95,
         'train/test split',
         ha='right', va='top', color='gray')
plt.title('Model Predictions: Train vs Test')
plt.xlabel('Date')
plt.ylabel('kWh')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Графік історії навчання (loss & mae)
plt.figure(figsize=(12,5))
# Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
# MAE
plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
