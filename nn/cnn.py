# CNN from scratch
import numpy as np

## We build convulution 2D 
# There are three eleemnt for a CNN neural network 
# Convoulution Layer , Pooling Layer and Fully Connected Layer
# 
# The difficult part would be how to do the back propagation with CNN  

class Conv2D:
    def __init__(self, input_channel: int | tuple, output_channel: int | tuple, kernel_size: int, stride: int = 0, padding: int = 0):
        self.kernel_size = kernel_size
        self.input_channel = input_channel
        self.padding = 0 
        self.stride = 1 
        self.weight = np.random.rand(output_channel, input_channel, kernel_size, kernel_size)

    def forward(self, x):
        N, C_in, H, W = x.shape
        out_h = int(np.floor((H + 2*self.padding - self.kernel_size) / self.stride) + 1)
        out_w = int(np.floor((W + 2*self.padding - self.kernel_size) / self.stride) + 1)
        
        x_unfold = unfold(x, self.kernel_size, self.stride, self.padding)
        # Assuming unfold returns (N, C_in, out_h, out_w, kernel_size, kernel_size)
        # Reshape to (N, C_in * kernel_size * kernel_size, out_h * out_w)
        x_unfold = x_unfold.reshape(N, C_in * self.kernel_size * self.kernel_size, out_h * out_w)
        
        weight_flat = self.weight.reshape(self.weight.shape[0], -1)
        
        output = np.einsum('oc,ncl->nol', weight_flat, x_unfold)
        
        output = output.reshape(N, self.weight.shape[0], out_h, out_w)
        return output
            




class MaxPool2D:
    def __init__(self, kernel_size: int, padding: int = 0, stride: int = 1):
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        '''
        Forward pass for max pooling.
        Args:
            x: Input tensor of shape (N, C, H, W)
        Returns:
            Output tensor of shape (N, C, out_h, out_w)
        '''
        # Extract patches using the corrected unfold function
        x_unfold = unfold(x, self.kernel_size, self.stride, self.padding)
     #   print(x_unfold.shape)
        # Compute max over the patch dimensions (kernel_size x kernel_size)
        r = np.max(x_unfold, axis=(4, 5))
        return r


def unfold(x, kernel_size, stride, padding):
    '''
    Unfold operation to extract patches from the input tensor for max pooling.
    Args:
        x: Input tensor of shape (N, C, H, W)
        kernel_size: Size of the pooling window
        stride: Stride of the sliding window
        padding: Padding applied to height and width
    Returns:
        Patches of shape (N, C, out_h, out_w, kernel_size, kernel_size)
    '''
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    N, C, H, W = x.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1
    
    # Define the shape of the output patches
    shape = (N, C, out_h, out_w, kernel_size, kernel_size)
    
    # Get the strides of the input tensor
    s0, s1, s2, s3 = x.strides
    # Strides for sliding the window: batch, channel, height stride, width stride, within-patch height, within-patch width
    strides = (s0, s1, s2 * stride, s3 * stride, s2, s3)
    
    # Extract patches
    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return patches
