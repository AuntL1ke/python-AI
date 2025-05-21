import numpy as np
import matplotlib.pyplot as plt


x = np.random.uniform(0,1,100)
y = np.random.uniform(0,1,100)


plt.scatter(x,y,color='green',alpha=0.6)

plt.xlabel('X values')
plt.ylabel('Y values')

plt.title('Scatter Plot of Uniformly Distributed Points')
plt.grid(True)
plt.show()


x = np.linspace(-10,10,500)

y1 = np.sin(x)
y2 = np.cos(x)
y3 = y1 + y2


plt.plot(x, y1, label='y = sin(x)', color='blue')
plt.plot(x, y2, label='y = cos(x)', color='blue')
plt.plot(x, y3, label='y = sin(x) + cos(x)', color='blue')


plt.legend()


plt.grid(True)

plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.title('Graphs of y = sin(x), y = cos(x), and y = sin(x) + cos(x)')

plt.show()