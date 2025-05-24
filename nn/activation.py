import numpy as np 

import numpy as np

class LeakyRelu:
    def __init__(self, negative_slope=1e-2):
        self.negative_slope = negative_slope
        self.x = None  # Store input for backward pass
        self.o = None  # Store output for consistency

    def forward(self, x):
        self.x = x  
        self.o = self._leaky(x)
        return self.o

    def _leaky(self, x):
        return np.where(x >= 0, x, x * self.negative_slope)

    def _grad(self, x):
        return np.where(x >= 0, 1, self.negative_slope)

    def backward(self, loss):
        """
        Given dl/do (gradient of loss w.r.t. output), compute dl/dx.
        Input:
            loss: Gradient of loss w.r.t. output, same shape as self.o
        Output:
            dl/dx: Gradient of loss w.r.t. input, same shape as self.x
        """
        return loss * self._grad(self.x)
    

class Softmax:
    def __init__(self):
        self.x = None
        self.o = None
        self.exp_x = None 
    
    def forward(self, x):
        """
            consider x: [N , input_dimension]
        """
        self.x = x
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        self.exp_x = np.exp(x_shifted)
        sum_exp_x = self.exp_x.sum(axis=1, keepdims=True)
        self.o = self.exp_x / sum_exp_x
        print(self.o)
        return self.o
    
    def backward(self ,loss):
        """
            loss is [N, input_dimension]
            we need to find :
            dl/do 
            dl/dx = dl/do . do/dx 
        """
        dot_product = np.sum(loss * self.o, axis=1, keepdims=True)
        dx = self.o * (loss - dot_product)
        return dx
    

class Tanh:
    pass