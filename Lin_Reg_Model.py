import numpy as np

class Linear_Regression():
    def __init__(self, Learning_Rate, no_of_iterations):
        self.Learning_Rate = Learning_Rate
        self.no_of_iterations = no_of_iterations
    
    def fit(self, X, Y):

        self.X = X
        self.Y = Y
        #initialize weights
        self.m, self.n = X.shape # m is no.of data points and n is no.of features
        self.w = np.zeros(self.n)
        self.b = 0

        for i in range(self.no_of_iterations):
            self.update_weights()
    
    def update_weights(self):

        Y_pred = self.predict(self.X)

        dw = -2 * (self.X.T).dot(self.Y - Y_pred) / self.m
        db = -2 * np.sum(self.Y - Y_pred) / self.m

        self.w = self.w - self.Learning_Rate * dw
        self.b = self.b - self.Learning_Rate * db
    
    def predict(self, X):
        return X.dot(self.w) + self.b