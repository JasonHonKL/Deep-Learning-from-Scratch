# CNN from scratch
import numpy as np

## We build convulution 2D 
# Alex Net
class CNN:
    def __init__(self):
        self.conv1 = Conv2D()
        self.relu1 = LeakyRELU()
        self.pool1 = MaxPool2D()
        self.conv2 = Conv2D()
        self.relu2 = LeakyRELU()
        self.pool2 = MaxPool2D()
        self.conv3 = Conv2D()
        self.conv4 = Conv2D()
        self.conv5 = Conv2D()
        self.relu3 = LeakyRELU()
        self.pool3 = MaxPool2D()
        self.fc1 = Linear()
        self.fc2 = Linear()
        self.fc3 = Linear()

# There are three eleemnt for a CNN neural network 
# Convoulution Layer , Pooling Layer and Fully Connected Layer
# 
# The difficult part would be how to do the back propagation with CNN  

class Conv2D:
    """
        Expected Outpu Dimension: 
        H_out: floor(H_in + 2 x padding[0] - dilation[0] x (kernel[0] -1) -1)/ stride[0]) +  1
        W_out: floor(W_in + 2 x padding[0] -dilation[0] x (kernel[0] -1)-1)/stride[0]) + 1 
    """
    def __init__(self ,input_channel:int|tuple , output_channel:int|tuple , kernel_size:int , stride:int = 0 , padding:int = 0):
        self.kernel_size = kernel_size
        self.input_channel = input_channel

        self.weight = np.random.rand(output_channel , input_channel , kernel_size , kernel_size)

    """
        N: batch size 
        C_in: Number of channel
        H: height input 
        W: width input 
    """
    def forward(self , x):
        """
            x_shape = N x c_in x H x W 
        """

        pass 


class MaxPool2D:
    pass 

class Linear:
    pass 

class LeakyRELU:
    pass 

class Softmax:
    pass 

