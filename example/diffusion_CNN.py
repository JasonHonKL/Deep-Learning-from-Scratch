import numpy as np 

from nn.cnn import Conv2D , MaxPool2D 
from nn.mlp import Linear 



class CNN:
    """
        Diffusion models taks 
        input: [n, 2]
        output:[n, 2]
    """
    def __init__(self):
        """
            We need to use Conv1D 
        """
        pass 
    

    def forward(self , x , i):
        pass


class DiffusionCNN:
    def __init__(self ,beta):
        """
            Diffusion CNN is an example how we could do diffusion with the intermediate layer as CNN
            Here we assume that beta is a constant

            This diffusion takes  (N x 2) input output (N x 2) outpup
        """
        self.gamma = 1  
        self.beta = beta 
        self.nn = CNN()


    def forward(self , x , t):
        gamma_t = (1- self.beta)**(t) 
        epsilon = np.random.normal(size=x.shape)
        x_t = np.sqrt(gamma_t) * x + np.sqrt(1-gamma_t) * epsilon
        return x_t 

    def sampling(self , t ):
        # we make 100 random sample noise
        x = np.random.normal(size=(10 , 2))

        print(1- self.beta)
        for i in range(t):
            beta_t = self.beta 
            alpha_t = 1- beta_t
            alpha_bar_t = alpha_t ** i
            epsilon_theta = CNN().forward(x , i)
            mean = (x - (beta_t / np.sqrt(1 - alpha_bar_t)) * epsilon_theta) / np.sqrt(alpha_t)
            std = np.sqrt(beta_t)
            z = np.random.normal(0 , 1 , x.shape) if i > 1 else np.zeros_like(x)

            x = mean + std*z 

        return x

    def backward(self , loss):
        pass

    def fit(self):
        pass


## testing purpose only 
# TODO: should be removed when finish DIffusionCNN

diff = DiffusionCNN(0.1)
## see how it beocmes random noise
for i in range(0 , 1000 , 100):
    result = diff.forward(np.array([[1,2] , [10,20]]) , i)
    print(f"result:{i} : \n {result}")

sample = diff.sampling(100)
#print(sample)