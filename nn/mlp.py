import numpy as np

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