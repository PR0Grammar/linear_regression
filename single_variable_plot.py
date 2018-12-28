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
lin_reg.gradient_descent()

predict_one = np.matmul(np.array([1, 15.0]), lin_reg.thetas)
predict_two = np.matmul(np.array([1, 13.0]), lin_reg.thetas)
predict_three = np.matmul(np.array([1, 4.7]), lin_reg.thetas)


print('Prediction for x = 15.0: ' + str(predict_one[0]))
print('Prediction for x = 13.0 ' + str(predict_two[0]))
print('Prediction for x = 4.7 ' + str(predict_three[0]))