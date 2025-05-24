# Deep Learning from scretch
This project aims build Nerual Netowrk netowrk from scretch with only numpy. 
Build your own deep learning network is the simpliest way to learn what is deep learning and neural network. Unlike other subjectes ,currently there are limited resources/book exercies for us to understand nerual network. This repo is built for myself to build everything from scretech such that I could understand how it works behind. 

This is far from efficienct compare to how it done in pytorch, yet I do think it is far enough for me to get used to the logic behind deep learning.

However this library is totally not for the purpose of production. It is just for us to learn about the mechanism behind. For deail and production usage one should use a more well developed library ! 

If this repo can help you to learn don't forget to give me a little star. 

Todo List:
- [x] ANN / MLP 
- [ ] CNN
- [x] Diffusion with MLP
- [x] Transformer 

    - [ ] Vision Transformer (if available)
- [ ] RNN & LSTM

## How to use this library 

### Linear
To use linear layer we can call
```py
from nn.linear import Linear
import numpy as np

linear = Linear(input_dim=2, output_dim=1)

x = np.array([[1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0]])

y_true = np.array([[1.0], [2.0], [3.0]])

y_pred = linear.forward(x)

loss = np.mean((y_pred - y_true) ** 2)
grad_loss_wrt_output = 2 * (y_pred - y_true) / y_true.shape[0]

print(f"Loss before update: {loss:.4f}")

grad_w, grad_x = linear.backward(grad_loss_wrt_output)

learning_rate = 0.01
linear.weight -= learning_rate * grad_w

y_pred_after = linear.forward(x)
loss_after = np.mean((y_pred_after - y_true) ** 2)
print(f"Loss after update: {loss_after:.4f}")

```

See the [/example/mlp.py](mlp.py) for MLP example using linear. 