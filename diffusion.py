# Diffusion
import numpy as np 
# Before we start to write our own diffusion we have to understand what is diffusion

# There are two step for a diffusion model
# One is forward let f: x --> x + epsilon
# Another one is backward b: x,t -->epsilon, x := x - epsilon
# Some tricks in forward processing will be mentioned

"""
    Note diffusion is not a nerual architecture instead it is telling the network
    what the learn. It is a genreative method or in more simple terms it is a simpling method

    e.g) we want to sample in a 2D space 
    
    sample (x, y) --> go through t time backward --> (correct x ,y)
    so our job would be 
    build a NN to learn from (x,y) --> (correct x ,y)
    
"""

"""
    As usual same, we first build a linear nn ourself as a revision 
"""
class Linear:
    def __init__(self , input_dimension:int=1 , output_dimension=1):
        """
            usage: W dot x ==> (input , output) . (N , input) --> (N , output)
        """
        self.weight = np.random.randn(input_dimension , output_dimension)
        self.bias = np.random.randn(output_dimension)

    def forward(self,  x):
        """
        
        """
        self.output = np.einsum('io,ni->no', self.weight, x) + self.bias
        self.gradient = x 
        return self.output
    
    def backward(self , loss):
        """
            This is the backward pass 
            loss should be the gradient from previous round 
            back ward pass would return the gradient till here and 
            Here's some notes for your information

            partial L / partial x ==> 

        """
        grad_w = self.gradient.T @ loss
        grad_input = loss @ self.weight.T
        grad_bias = loss.sum(axis=0)
        learning_rate = 0.01
        self.weight -= learning_rate * grad_w
        self.bias -= learning_rate * grad_bias.T

        return grad_input # return \partial W / \partial X

        
class Sigmoid:
    def __init__(self):
        pass
    
    def forward(self ,x):
        self.output = self._sig(x)
        return self.output

    def backward(self , loss):
        sigmoid_derivative = self.output * (1 - self.output)
        grad = loss * sigmoid_derivative
        return grad

    def _sig(self , x):
        return 1 /(1 + np.exp(-x))


class MLP:
    def __init__(self):
        self.l1 = Linear(2, 512)
        self.activation = Sigmoid()
        self.l2 = Linear(512, 1)

    def forward(self, x):
        a1 = self.l1.forward(x)                # First linear layer
        z1 = self.activation.forward(a1)       # Activation after first linear
        a2 = self.l2.forward(z1)               # Second linear layer
       # z2 = self.activation.forward(a2)       # Activation after second linear (output)
        return a2

    def backward(self, loss):
     #   g = self.activation.backward(loss)    # Backprop through output activation
        g = self.l2.backward(loss)                # Backprop through second linear
        g = self.activation.backward(g)       # Backprop through first activation
        g = self.l1.backward(g)                # Backprop through first linear
        return g


class Diffusion:
    """
        Out diffusion takes 
        (x , y) --> (x ,y)
    """
    def __init__(self):
        """
            Our model would take input (x , t) --> (y) it would be any x , t combination
            the y is random sampling , i.e it is normally distriabled with x as the mean with 1 std
        """

        # input dimension should be 2 because 
        self.l1 = Linear(2 , 4)
        self.a1 = Sigmoid()
        self.l2 = Linear(4 ,8)
        self.a2 = Sigmoid()
        self.l3 = Linear(8 ,16)
        self.a3 = Sigmoid()
        self.l4 = Linear(16 ,1)

        self.beta = 0.1
        self.gamma = 1 - self.beta # assume constant beta

    def forward(self , x , t):
        """
            x at time t: ==> x_t = sqrt(gamma_t)x_0 + sqrt(1-gamma_t) epsilon
        """
        gamma_t = 1.0
        for i in range(t):
            gamma_t *= self.gamma  # (1 - beta)^t

        epsilon = np.random.normal(0 , 1 , x.shape)
        x_t = np.sqrt(gamma_t)*x + np.sqrt(1 - gamma_t)*epsilon
        return x_t , gamma_t

    def backward(self, x , t):
        """
            Here we will need to use the nerual netowrk under the diffusion 
        """

        # we want to predict the noise here
        for _ in range(t):
            pass

    def train():
        pass