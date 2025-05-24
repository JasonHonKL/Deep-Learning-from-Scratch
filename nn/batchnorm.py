# This file aims to provide library related to batch normalization

import numpy as np
"""
Reference: 
    https://arxiv.org/pdf/1502.03167

    - Avoid dropout problem
    - Address internal covariate shift
"""
class BatchNorm1D:
    def __init__(self , num_features):
        self.num_features = num_features
        self.epsilon = 0.0001 
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.y = 0
        self.x = None # TODO: safety check when calling forward

    def forward(self, x):
        """
            shape of x (n , d)
            shape of output (n,  d) 
        """
        self.x = x
        self.mean = x.mean(axis=0)
        self.var = x.var(axis=0)
        self.x_normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        y = self.gamma * self.x_normalized + self.beta
        return y

    def backward(self , loss):
        """
            loss representing partial l / partial y ==> dimension [ n , num_offeatures ]
            we want to find:
                partial l / partial x
                partial l / partial gamma
                partial l / partial mu
                partial l / partial sigma
                partial l / partial x
        """
        dgamma = np.sum(loss * self.x_normalized, axis=0)
        dbeta = np.sum(loss, axis=0)

        # Gradient with respect to normalized input
        d_x_hat = loss * self.gamma

        # Standard deviation
        std = np.sqrt(self.var + self.epsilon)

        # Gradient with respect to input x
        mean_d_x_hat = np.mean(d_x_hat, axis=0)
        mean_d_x_hat_x_normalized = np.mean(d_x_hat * self.x_normalized, axis=0)
        dx = (d_x_hat - mean_d_x_hat - self.x_normalized * mean_d_x_hat_x_normalized) / std

        return dx, dgamma, dbeta

class BatchNorm2D:
    pass