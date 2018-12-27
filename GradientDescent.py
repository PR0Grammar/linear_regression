import numpy as np

class GradientDescent:
    def __init__(self, X, y, alpha = 0.01):
        self.alpha = alpha
        self.m = X.shape[0]
        self.n = X.shape[1] + 1
        self.thetas = np.zeros((X.shape[1] + 1, 1))
        self.features = self.features_init(X)
        self.results = y

    def features_init(self, features):
        theta_zero = np.ones((self.m, 1))
        return np.hstack((theta_zero, features))
    
    # Cost function for our prediction ( J(Theta) )
    def compute_cost(self):
        theta_transposed = self.thetas.transpose()
        prediction_err = 0
        for i in range(0, self.m):
            x = self.features[i: i + 1, 0:self.n]
            x = x.transpose()
            predicted_val = np.matmul(theta_transposed, x)[0, 0]
            prediction_err = prediction_err + pow((predicted_val - self.results[i, 0]), 2)
        
        prediction_err = (1.0 / (2.0 * self.m)) * prediction_err
        return prediction_err
