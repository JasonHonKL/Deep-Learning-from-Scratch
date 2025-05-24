"""
    TODO: Refactor move linear layer to here
"""
import numpy as np

class Linear:
    def __init__(self , input_dim , output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.randn(input_dim , output_dim)
        self.bias = np.zeros(shape=(output_dim))
        self.x = None
        self.o = None
    
    def forward(self ,x):
        self.x =x 
        self.o = np.einsum('ni,ij->nj' , x , self.weight) +self.bias
        return self.o


    def backward(self ,loss):
        """
            loss dimensiont: [N , o]

            backward return dl/dw , dl/dx in order
            dl/dw = dl/do * do/dw = dl/do * x
            dl/dx = dl/do * do/dx = dl/do * w 

        """
        dw = np.einsum('no, ni -> io' , loss , self.x)
        dx = np.einsum('no, oj -> nj ' , loss , self.weight.T)
        db = np.sum(loss, axis=0)
        return dw , dx , db

