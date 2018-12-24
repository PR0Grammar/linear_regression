import numpy as np
import matplotlib.pyplot as plt

data_matrix = np.loadtxt('./exdata1.txt')

x = data_matrix[:, 0]
y = data_matrix[:, 1]

plt.plot(x, y, 'ro')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.plot()
plt.show()
