import numpy as np
import pandas as pd

# Кількість точок — одна кожні 1.44 хв (щоб вийшло рівно 1000 точок за добу)
n = 10000
departure_minutes = np.linspace(0, 1439, n, dtype=int)

def realistic_trip_duration_deterministic(m):
    base = 20

    if 420 <= m <= 540:  # Ранковий пік
        peak = 10 * np.exp(-((m - 480) / 30) ** 2)  # Зменшена амплітуда
    elif 990 <= m <= 1110:  # Вечірній пік
        peak = 12 * np.exp(-((m - 1050) / 30) ** 2)  # Зменшена амплітуда
    elif 720 <= m <= 840:  # Обідній спад
        peak = -1.5 * np.exp(-((m - 780) / 20) ** 2)  # Меньша амплітуда спаду
    elif 0 <= m <= 300:  # Глибока ніч
        peak = -1  # Легкий спад у глибоку ніч
    else:
        peak = 0

    return round(base + peak, 2)

# Генерація тривалості по часах відправлення
trip_duration = np.array([realistic_trip_duration_deterministic(m) for m in departure_minutes])
departure_time = [f"{m // 60:02d}:{m % 60:02d}" for m in departure_minutes]

# Створення DataFrame
df = pd.DataFrame({
    "departure_time": departure_time,
    "departure_minutes": departure_minutes,
    "trip_duration_min": trip_duration
})

# Збереження до CSV
df.to_csv("trip_duration.csv", index=False)

# Показати кілька перших рядків
df.head()