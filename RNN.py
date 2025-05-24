import numpy as np 


class RNN:
    def __init__(self , input_dim , hidden_dim , output_dim):
        self.hidden_dim = hidden_dim

        # three weight matrix
        self.Wxh = np.random.randn(input_dim , output_dim)
        self.Whh = np.random.randn(hidden_dim , hidden_dim)
        self.Why = np.random.randn(hidden_dim , output_dim)
