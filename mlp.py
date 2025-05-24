import numpy as np

class ANN:
    def __init__(self, g=False):
        self.l1 = Linear(2, 1024)
        self.activation = Sigmoid()
        self.l2 = Linear(1024, 1)
        self.output = None
        self.g = g
    
    def __call__(self, x):
        self.output = self.forward(x)
        return self.output

    def forward(self, x):
        self.x = x
        self.a1 = self.l1.forward(x)  # x is (batch_size, 2), a1 is (batch_size, 4)
        self.z1 = self.activation(self.a1)  # z1 is (batch_size, 4)
        self.output = self.l2.forward(self.z1)  # output is (batch_size, 1)
        return self.output
    
    def backward(self, loss):
        g = loss.backward()
        g = self.l2.backward(self.z1, g)
        g = self.activation.backward(self.a1, g)
        g = self.l1.backward(self.x, g)

    def zero_grad(self):
        self.l1.zero_grad()
        self.l2.zero_grad()
    
class Linear:
    """
        Expected input: batch_size x n dimension vector, output: batch_size x m dimension vector
        Wx + b = y => dimension of xW + b => (batch_size x n) . (n x m) + (m,) => (batch_size x m) dimensional vector 
        n stands for input dimension, m stands for output dimension
    """
    def __init__(self, n: int, m: int):
        self.weight = np.random.rand(n, m)  # Random weights: (n, m)
        self.bias = np.zeros(m)            # Bias initialized to zeros: (m,)
        self.gradient = np.zeros((n, m))   # Weight gradient: (n, m)
        self.bias_gradient = np.zeros(m)   # Bias gradient: (m,)
        self.m = m 
        self.n = n
    
    def zero_grad(self):
        self.gradient = np.zeros((self.n, self.m))
        self.bias_gradient = np.zeros(self.m)  # Reset bias gradient

    def forward(self, x):
        """
        Expect x to be a batch_size x n vector
        Returns batch_size x m vector after applying weights and bias
        """
        return np.dot(x, self.weight) + self.bias

    def backward(self, x, g):
        """
        Compute gradients for weights and biases, update parameters, and return gradient w.r.t input
        x: input of shape (batch_size, n)
        g: gradient from the next layer of shape (batch_size, m)
        """
        # Ensure x and g have proper shapes
        if len(x.shape) == 1:
            x = x.reshape(1, -1)  # Reshape to (1, n) for single sample
        if len(g.shape) == 1:
            g = g.reshape(1, -1)  # Reshape to (1, m) for single sample

        # Compute gradients averaged over the batch
        self.gradient = np.dot(x.T, g) / x.shape[0]      # Weight gradient: (n, m)
        self.bias_gradient = np.mean(g, axis=0)          # Bias gradient: (m,)
        learning_rate = 0.01  # Fixed learning rate, consistent with original
        self.weight -= learning_rate * self.gradient     # Update weights
        self.bias -= learning_rate * self.bias_gradient  # Update biases
        grad_input = np.dot(g, self.weight.T)            # Gradient w.r.t input: (batch_size, n)
        return grad_input
        

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
    def __call__(self,x):
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


class MSELoss:
    def __init__(self):
        pass

    def __call__(self, predict, output):
        self.predict = predict
        # Reshape output to match predict's shape
        if len(output.shape) == 1:
            output = output.reshape(-1, 1)
        elif output.shape[0] == 1:
            output = output.T  # Convert (1, N) to (N, 1)
        self.output = output
        self.loss = ((self.predict - self.output) ** 2) * 0.5
        return self.loss

    def backward(self):
        # Gradient of MSE loss w.r.t predictions: (predict - output)
        gradient = (self.predict - self.output)
        return gradient


# may be also add dropout here