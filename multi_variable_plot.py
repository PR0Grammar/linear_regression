import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import interpolate
from LinearRegression import LinearRegression


data_matrix = np.loadtxt('./ex2data.txt')

x = data_matrix[0: data_matrix.shape[0], 0: 2]
y = data_matrix[0: data_matrix.shape[0], 2: 3]

lin_reg = LinearRegression(x, y)

lin_reg.normal_equation()

# Graph for our data set

X = x[:, 0]
Y = x[:, 1]
Z = y[:, 0]

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(X, Y, Z, c='r', marker='o')

# Our predicted values (hypothesis function)
predicted_val = []
for i in range(0, data_matrix.shape[0]):
    predicted_val.append(np.matmul([1, x[i, 0], x[i, 1]],lin_reg.thetas)[0])
predicted_val = np.array(predicted_val)
ax.plot_trisurf(X, Y, predicted_val, color='blue')

plt.show()