import numpy as np 

from nn.activation import Tanh

class RNN:
    def __init__(self , input_dim , hidden_dim , output_dim):
        self.hidden_dim = hidden_dim

        # three weight matrix
        self.Wxh = np.random.randn(input_dim , output_dim)
        self.Whh = np.random.randn(hidden_dim , hidden_dim)
        self.Why = np.random.randn(hidden_dim , output_dim)

        # two bias matrix
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)
    
    def forward(self , x , h_prev):
        """
            In RNN we would take two parameters
        """
        self.h_t = Tanh.forward( np.matmul(x , self.Wxh) + np.matmul(h_prev , self.Whh) + self.b_h) 
        self.y_t = np.matmul(self.h_t , self.Why) + self.b_y
        return self.h_t , self.y_t
    
    def backward(self , loss):
        """
            loss is MSE from y / any other type of loss from y
            ==> dimension [n , O]
            ==> expected return dl/dxh , dl/dwhh , dl/dwhy 
            we have to get every thing in order to update the weight matrix and the bias matrix
        """

        pass 

