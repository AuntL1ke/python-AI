import pandas as pd
import matplotlib.pyplot as plt


data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 40],
    'Salary': [50000, 60000, 70000, 80000],
    'Experience': [1, 3, 5, 7]
}

df = pd.DataFrame(data)

print(df)



average_salary = df['Salary'].mean()
print("Average salary:", average_salary)


std_salary = df['Salary'].std()
print("Standard deviation:", std_salary)


min_age = df['Age'].min()
max_age = df['Age'].max()
print("Min age:", min_age)
print("Max age:", max_age)



plt.figure(figsize=(8,5))
plt.plot(df['Age'],df['Salary'], marker='o', linestyle='--')
plt.title('Dependency salary from age')
plt.xlabel("Age")
plt.ylabel("Salary")
plt.grid(True)
plt.show()