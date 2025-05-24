# test CNN
from nn.cnn import Conv2D , unfold , MaxPool2D
import numpy as np

# Test Conv2D
conv2d = Conv2D(1 , 3 , 2)
maxPool = MaxPool2D(2)

sample_input = np.array([[[[1 ,2 , 3] , [3, 4 ,5] , [6 ,7 ,8] , [7 , 8 ,9]]]])
r = conv2d.forward(sample_input) 


sample_input = np.array([[
    # Channel 1
    [
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9,  10, 11, 12],
        [13, 14, 15, 16]
    ],
    # Channel 2
    [
        [2,  4,  6,  8],
        [10, 12, 14, 16],
        [18, 20, 22, 24],
        [26, 28, 30, 32]
    ]
]])
maxPool.forward(sample_input)

#sample_input = np.random.randn(3 , 2, 6 ,6)
#conv2d.forward(sample_input)



