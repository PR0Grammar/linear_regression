import numpy as np
import matplotlib.pyplot as plt
from GradientDescent import GradientDescent

data_matrix = np.loadtxt('./exdata1.txt')

x = data_matrix[0:data_matrix.shape[0], 0:1]
y = data_matrix[0:data_matrix.shape[0], 1:2]


lin_reg = GradientDescent(x, y) 
lin_reg.gradient_descent()

predicted_x = []
predicted_y = []

i = 0;

while(i < 23.50):
    predicted_x.append(i)
    hypoth_of_i = np.matmul([1, i], lin_reg.thetas)[0]
    predicted_y.append(hypoth_of_i)
    i = i + 0.5
plt.plot(x, y, 'ro')
plt.plot(predicted_x, predicted_y)
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.plot()
plt.show()