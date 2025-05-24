import numpy as np 

class LekyRelu():
    def __init__(self , negative_slope = 1e-2):
        self.negative_slope = negative_slope

    def forward(self, x):
        self.o = self._leky(x)
        return self.o

    def _leky(self, x):
        return np.where(x >= 0, x, x * self.negative_slope)

    def _grad(self , x):
        return np.where(x>= 0, 1 , self.negative_slope)

    def backward(self , loss):
        """
            given dl/do  [n , ]
            return dl/dx = dl/do * do/dx
        """
        return self._grad(loss)


l = LekyRelu()
s = np.array([[1,2,3] , [-2,3,4]])
y = np.array([[2 ,1 ,4] ,[1 ,3, 4] ])
loss = s - y
k = (l.forward(s))
print(l.backward(k))