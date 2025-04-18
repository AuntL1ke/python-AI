import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("energy_usage.csv")
print(df.head())

X = df[['temperature', 'humidity', 'hour', 'is_weekend']]
y = df['consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

my_data = pd.DataFrame([{
    'temperature': 20,
    'humidity': 63,
    'hour': 20,
    'is_weekend': 0
}])

predicted_consumption = model.predict(my_data)

print(f"Прогнозоване споживання: {predicted_consumption[0]:,.2f} кВт/год")

y_pred = model.predict(X_test)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape:.2f}%")

plt.scatter(y_test, y_pred)
plt.xlabel("Справжне споживання")
plt.ylabel("Прогнозоване споживання")
plt.title("Справжнє vs Прогнозоване споживання")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()