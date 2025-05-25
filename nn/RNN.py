import numpy as np 

from nn.activation import Tanh , Sigmoid

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

class LSTM:
    """
        a simple RNN would takes an input and the previous hidden state
        then it would output a new hidden state 

        -- it is simple but would forget previous information due to vanishing gradients

        LSTM: 
            introduce cell state and 
                - forget gate: decides what to discard from the previous cell state 
                - input state: determines what new information to add to the cell state.
                - output gate: controls what to output based on the current cell state.
                - cell memory: A "conveyor belt" that carries information across time step

    """
    def __init__(self , input_dim , output_dim , output_size , lr=0.01):
        """
            Initialize the LSTM with input and output dimensions:

            Args:
                input_dim (int): Size of the input vector
                output_dim (int): size of the hidden state

            Expected:
                Initalize all weight values
        """
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.output_size = output_size
        self.lr = lr 

        # input [h_prev , x]  --> [h_prev +x]
        concate_dim = self.hidden_dim + self.input_dim
        
        # forget gate 
        self.Wf = np.random.randn(self.hidden_dim , concate_dim) * 0.01
        self.bf = np.zeros(shape=(self.hidden_dim , 1))

        self.Wi = np.random.randn(self.hidden_dim  , concate_dim) * 0.01
        self.bi = np.zeros((self.hidden_dim , 1))

        self.Wc = np.random.randn(self.hidden_dim , concate_dim)*0.01
        self.bc = np.zeros((self.hidden_dim , 1))

        self.Wo = np.random.randn(self.hidden_dim , concate_dim) * 0.01
        self.bo = np.zeros((self.hidden_dim , 1))

        # Initialize weight for output layer
        self.W_out = np.random.randn(self.output_size , self.hidden_dim) * 0.01
        self.b_out = np.random.zeros((self.output_size))

        # set h_0 and c_0
        self.h = np.zeros((self.hidden_dim , 1))
        self.c = np.zeros((self.hidden_dim , 1))

    def forward(self , x):
        """
            forward pass for one time step
        """
        concact = np.vstack((self.h , x))
        
        # forget gate 
        f = Sigmoid().forward(np.dot(self.Wf , concact) + self.bf)
        
        # input gate
        i = Sigmoid().forward(np.dot(self.Wi , concact) + self.bi)

        c_tilde = Tanh().forward(np.dot(self.Wc , concact) + self.bc)

        # forget + new info
        self.c = f * self.c + i * c_tilde

        o = Sigmoid().forward(np.dot(self.Wo , concact) + self.bo) # we should return this ? 

        self.h = o * np.tanh(self.c)
        # we have to do some chaching here 
        y_t = np.dot(self.W_out , self.h) + self.b_out
        return self.h 


    def backward(self ,loss):
        """
            Args:
                loss: dl/dy

            backward: 
                we have to find 
                    dl/dw_f 
                    dl/dw_o
                    dl/dw_c
        """
        # first we have to find dl/dh = dl/dy * dy/dh 

        #TODO backpropagation to find the gradient