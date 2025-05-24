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
    

class Sigmoid:
    """
        This function aims to provide a sigmoid class for the puspose of non linear layer 
        input is a nxm array 
        Output Dimension nxm array 
    """
    def __init__(self):
        self.gradient = 0

    def zero_grad(self):
        self.gradient = 0 
    
    def sigmoid(self ,x):
        return 1/(1+ np.exp(-x))
    """
        Call the sigmod 
        given x it will elementwise run the vector multiplication 
    """
    def forward(self,x):
        vectorized_sigmoid = np.vectorize(self.sigmoid)
        return vectorized_sigmoid(x) 
    
    """
        Backward pass: given the forward x --> calculate the gradient give gradient from above layer 
        d sigmoid = (1- sigmoid)(sigmoid)
    """
    def backward(self, x, g):
        # Compute sigmoid output for the forward pass
        sig = self.sigmoid(x)
        # Gradient through sigmoid: g * sigmoid(x) * (1 - sigmoid(x))
        grad = g * sig * (1 - sig)
        return grad


class Tanh:
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self,  d_out):
        return d_out * (1- self.output**2)