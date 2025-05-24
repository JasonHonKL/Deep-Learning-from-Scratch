# this file provided as an example how to use mytinypytorch to run a mlp

from nn.linear import Linear
from nn.activation import LeakyRelu
from  nn.mlp import MSELoss
import numpy as np

class MLP:
    def __init__(self):
        self.l1 = Linear(1 , 2)
        self.relu1 = LeakyRelu()
        self.l2 = Linear(2 , 1)
        
        self.a1 = None 
        self.z1 = None
        self.a2 = None

        self.lr = 0.0001
        self.loss =MSELoss()

    def forward(self, x):
        self.a1 = self.l1.forward(x)
        self.z1 =self.relu1.forward(self.a1)
        self.a2 = self.l2.forward(self.z1)
        return self.a2

    def fit(self, x, y , epoch = 1 ):
        """
            Assumption:
                x dimension: [n , input]
        """
        for _ in range(epoch):
            pred = self.forward(x)

            self.loss(pred , y)
            grad = self.loss.backward()
            dw , dx , db =  self.l2.backward(grad)
            self.l2.weight -= self.lr * dw
            self.l2.bias -= self.lr * db

            grad = self.relu1.backward(dx)
            dw , dx , db = self.l1.backward(grad)
            self.l1.weight -= self.lr*dw
            self.l1.bias -= self.lr * db

    