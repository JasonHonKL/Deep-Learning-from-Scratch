# This file aims to provide library related to batch normalization

import numpy as np
"""
Reference: 
    https://arxiv.org/pdf/1502.03167

    - Avoid dropout problem
    - Address internal covariate shift
"""
class BatchNorm1D():
    def __init__(self , num_features):
        self.num_features = num_features
        self.epsilon = 0.0001 
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.y = 0

    def forward(self, x):
        """
            shape of x (n , d)
            shape of output (n,  d) 
        """
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
        self.y = self.gamma * x_normalized + self.beta
        return self.y

    def backward(self , loss):
        pass


bn = BatchNorm1D(num_features=2)

# Sample input: 3 samples, 2 features
x = np.array([[1, 2], [3, 4], [5, 6]])

# Apply batch normalization
y = bn.forward(x)
print("Normalized output:")
print(y)