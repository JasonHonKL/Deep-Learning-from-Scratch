import numpy as np 

class Linear:
    def __init__(self , input_dim , outpu_dim):
        self.input_dim = input_dim
        self.output_dim = outpu_dim

        self.W = np.random.randn(0 , 0.01 , (input_dim , outpu_dim))
        self.b = np.zeros(outpu_dim)

    def forward(self , x):
        """
            x-dimension: (N , i)
            w-dimension: (i , o)
            output dimension => (N , o)
        """
        self.input = x
        return np.dot(x ,self.W) + self.b

    def backward(self, d_out):
        d_input = np.dot(d_out , self.W.T)
        d_W = np.dot(self.input.T , d_out)
        d_b = np.sum(d_out , axis=0)
        return d_input , d_W , d_b


class Tanh:
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self,  d_out):
        return d_out * (1- self.output**2)


class RNN:
    def __init__(self , input_dim , hidden_dim , output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear_xh = Linear(input_dim , hidden_dim)


