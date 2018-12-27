import numpy as np
import matplotlib.pyplot as plt
from GradientDescent import GradientDescent

data_matrix = np.loadtxt('./exdata1.txt')

x = data_matrix[0:data_matrix.shape[0], 0:1]
y = data_matrix[0:data_matrix.shape[0], 1:2]

plt.plot(x, y, 'ro')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.plot()
plt.show()

lin_reg = GradientDescent(x, y)

