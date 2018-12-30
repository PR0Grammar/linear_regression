import numpy as np

class LinearRegression:
    def __init__(self, X, y, alpha = 0.01, iterations = 1500):
        self.alpha = alpha
        self.iterations = iterations
        self.m = X.shape[0]
        self.n = X.shape[1] + 1
        self.thetas = np.zeros((X.shape[1] + 1, 1))
        self.features = self.features_init(X)
        self.results = y
        self.mean = None
        self.std_dev = None

    def features_init(self, features):
        theta_zero = np.ones((self.m, 1))
        return np.hstack((theta_zero, features))
    
    # Cost function for our prediction ( J(Theta) )
    def compute_cost(self):
        theta_transposed = self.thetas.transpose()
        prediction_err = 0
        for i in range(0, self.m):
            x = self.features[i: i + 1, 0: self.n]
            x = x.transpose()
            predicted_val = np.matmul(theta_transposed, x)[0, 0]
            prediction_err = prediction_err + pow((predicted_val - self.results[i, 0]), 2)
        
        prediction_err = (1.0 / (2.0 * self.m)) * prediction_err
        return prediction_err

    def normal_equation(self):
        features = self.features
        features_transposed = features.transpose()
        y = self.results

        self.thetas = np.dot(np.dot(np.linalg.inv((np.dot(features_transposed, features))), features_transposed), y)

    # Perform gradient descent to minimize cost
    def gradient_descent(self):
        # Matrix for all changes to the cost function with the corresponding theta values
        cost_function_changes = np.empty(( self.iterations, self.thetas.shape[0] + 1 )) 

        for i in range (0, self.iterations):
            # For each iteration, add the theta values with corresponding cost
            for m in range (0, self.thetas.shape[0]):
                cost_function_changes[i, m] = self.thetas[m, 0]
            cost_function_changes[i, cost_function_changes.shape[1] - 1] = self.compute_cost()

            updated_theta_values = []
            theta_transposed = self.thetas.transpose()

            # For each theta parameter, "descend" to a minimum cost using CURRENT theta values
            for j in range(0, self.n):
                theta_j = self.thetas[j, 0] 
                err_sum = 0
                # Computing the "summation" portion of our gradient descent
                for k in range(0, self.m):
                    x = self.features[k: k+1, 0: self.n]
                    x = x.transpose()
                    predicted_val = np.matmul(theta_transposed, x)[0,0]
                    predicted_val = (predicted_val - self.results[k, 0]) * self.features[k, j]
                    err_sum = err_sum + predicted_val

                new_theta_j = (self.alpha / self.m) * err_sum
                new_theta_j = theta_j - new_theta_j
                updated_theta_values.append(new_theta_j)

            # Simulatenous update of theta values
            for j in range(0, self.n):
                self.thetas[j, 0] = updated_theta_values[j]

        return cost_function_changes

    # Feature scales and mean normalize for features that have large "gaps" between values
    def feature_normalize(self):
        means = [1]
        std_dev = [1]
        for i in range(1, self.features.shape[1]):
            curr_feature = self.features[0: self.m, i: i + 1]
            feature_mean = np.mean(curr_feature)
            feature_standard_dev = np.std(curr_feature)

            self.features[:, i] = (self.features[:, i] - feature_mean) / feature_standard_dev
            means.append(feature_mean)
            std_dev.append(feature_standard_dev)
        self.mean = means
        self.std_dev = std_dev
    
    # If normalization used on training set, features used for prediction should be normalized as well
    def normalize_data(self, feature_values):
        if(self.mean is None or self.std_dev is None):
            print('Feature normalization not used on training set')
            return
        for i in range(0, len(feature_values)):
            feature_values[i] = (feature_values[i] - self.mean[i]) / self.std_dev[i]