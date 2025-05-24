import numpy as np
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
