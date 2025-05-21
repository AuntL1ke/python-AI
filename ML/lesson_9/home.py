import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Зчитування та предобробка даних ---
df = pd.read_csv("train.csv", usecols=[
    "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
])

# 1.1 Заповнюємо пропуски
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# 1.2 Категоріальні → числові
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# 1.3 Виділяємо X та y
X = df.drop("Survived", axis=1)
y = df["Survived"].astype(int)

# 1.4 Розділяємо на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1.5 Масштабування ознак
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --- 2. Побудова та компіляція моделі ---
model = keras.Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 3. Навчання з EarlyStopping ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=2
)

# --- 4. Прогноз на тестовій вибірці ---
y_pred_probs = model.predict(X_test).ravel()  # ймовірності класу "1"
y_pred_classes = (y_pred_probs > 0.5).astype(int)

# --- 5. Функції для візуалізації ---

def plot_training_curves(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Навчання & Валідація: точність')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Навчання & Валідація: втрата')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred_cls):
    cm = confusion_matrix(y_true, y_pred_cls)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Матриця неточностей')
    plt.show()


def plot_roc(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-крива')
    plt.legend()
    plt.grid(True)
    plt.show()


# --- 6. Відображаємо результати ---
plot_training_curves(history)
plot_confusion(y_test, y_pred_classes)
plot_roc(y_test, y_pred_probs)
