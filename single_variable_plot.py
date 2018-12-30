import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import interpolate
from LinearRegression import LinearRegression

data_matrix = np.loadtxt('./exdata1.txt')

x = data_matrix[0:data_matrix.shape[0], 0:1]
y = data_matrix[0:data_matrix.shape[0], 1:2]


lin_reg = LinearRegression(x, y) 
cost_function = lin_reg.gradient_descent()

predicted_x = []
predicted_y = []

i = 0;

while(i < 23.50):
    predicted_x.append(i)
    hypoth_of_i = np.matmul([1, i], lin_reg.thetas)[0]
    predicted_y.append(hypoth_of_i)
    i = i + 0.5

# Our data set with hypothesis function
data_set_graph = plt.figure()
plt.plot(x, y, 'ro')
plt.plot(predicted_x, predicted_y)
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')

# Our cost function
cost_function_graph = plt.figure()
ax = plt.axes(projection = '3d')

theta_zero = cost_function[:, 0]
theta_one = cost_function[:, 1]
cost = cost_function[:, 2]

X_1 = np.linspace(theta_zero.min(), theta_zero.max(), 400)
Y_1 = np.linspace(theta_one.min(), theta_one.max(), 400)
X_1_2d, Y_1_2d = np.meshgrid(X_1, Y_1)
Z_1 = interpolate.griddata((theta_zero, theta_one), cost,(X_1_2d, Y_1_2d), method='cubic')

ax.plot_wireframe(X_1, Y_1, Z_1, color='red')
ax.set_xlabel('Theta Zero')
ax.set_ylabel('Theta One')
ax.set_zlabel('Cost')

plt.show()

