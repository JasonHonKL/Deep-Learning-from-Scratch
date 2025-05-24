# Diffusion
import numpy as np 
# Before we start to write our own diffusion we have to understand what is diffusion

# There are two step for a diffusion model
# One is forward let f: x --> x + epsilon
# Another one is backward b: x,t -->epsilon, x := x - epsilon
# Some tricks in forward processing will be mentioned

"""
    Note diffusion is not a nerual architecture instead it is telling the network
    what the learn. It is a genreative method or in more simple terms it is a simpling method

    e.g) we want to sample in a 2D space 
    
    sample (x, y) --> go through t time backward --> (correct x ,y)
    so our job would be 
    build a NN to learn from (x,y) --> (correct x ,y)
    
"""


class Linear:
    def __init__(self, input_dimension:int=1, output_dimension:int=1):
        # weight: (in_dim, out_dim), bias: (out_dim,)
        self.weight = np.random.randn(input_dimension, output_dimension) * np.sqrt(2/(input_dimension+output_dimension))
        self.bias   = np.zeros(output_dimension)

    def forward(self, x):
        # x can be shape (in_dim,) or (N, in_dim)
        x = np.atleast_2d(x)                  # (1, in_dim) or (N, in_dim)
        self.input = x                       # cache for backward
        out = x.dot(self.weight) + self.bias # (N, out_dim)
        return out if out.shape[0]>1 else out[0]

    def backward(self, grad_output):
        # grad_output: (out_dim,) or (N, out_dim)
        grad = np.atleast_2d(grad_output)    # (N, out_dim)
        # accumulate gradients
        self.dW = self.input.T.dot(grad)     # (in_dim, out_dim)
        self.db = grad.sum(axis=0)           # (out_dim,)
        # gradient wrt inputs
        grad_input = grad.dot(self.weight.T) # (N, in_dim)
        return grad_input if grad_output.ndim>1 else grad_input[0]


class Sigmoid:
    def forward(self, x):
        self.out = 1/(1+np.exp(-x))
        return self.out

    def backward(self, grad_output):
        return grad_output * (self.out*(1-self.out))


class MLP:
    def __init__(self):
        self.l1 = Linear(2, 512)
        self.act = Sigmoid()
        self.l2 = Linear(512, 1)

    def forward(self, x):
        # x: shape (2,) or (N,2)
        a1 = self.l1.forward(x)
        z1 = self.act.forward(a1)
        a2 = self.l2.forward(z1)
        return a2

    def backward(self, grad_output):
        # grad_output: shape (1,) or (N,1)
        g2 = self.l2.backward(grad_output)
        g1 = self.act.backward(g2)
        _  = self.l1.backward(g1)
        # no return needed beyond
        

class Diffusion:
    def __init__(self, beta=0.1):
        self.mlp   = MLP()
        self.beta  = beta
        self.gamma = 1 - beta

    def forward_diffuse(self, x0, t):
        # x0: scalar or array
        gamma_t = self.gamma**t
        eps     = np.random.randn(*np.shape(x0))
        x_t     = np.sqrt(gamma_t)*x0 + np.sqrt(1-gamma_t)*eps
        return x_t, eps, gamma_t

    def backward_step(self, x_t, t):
        # predict noise
        eps_pred = self.mlp.forward(np.stack([x_t, np.full_like(x_t, t)], axis=-1))
        gamma_t  = self.gamma**t
        beta_t   = self.beta

        # reverse-mean formula
        mean = (1/np.sqrt(1-beta_t)) * (x_t - (beta_t/np.sqrt(1-gamma_t)) * eps_pred)
        if t > 1:
            noise = np.random.randn(*x_t.shape)
        else:
            noise = 0
        return mean + np.sqrt(beta_t)*noise

    def sample(self, T, shape):
        x = np.random.randn(*shape)
        for t in range(T, 0, -1):
            x = self.backward_step(x, t)
        return x

    def train(self, data, epochs=100, T=50, lr=1e-3):
        n = data.shape[0]
        for ep in range(1, epochs+1):
            perm, total_loss = np.random.permutation(n), 0.0
            for i in perm:
                x0 = data[i]
                # pick t, diffuse
                t = np.random.randint(1, T+1)
                x_t, eps, _ = self.forward_diffuse(x0, t)

                # predict and loss
                inp      = np.array([x_t, t])
                eps_pred = self.mlp.forward(inp)
                loss     = (eps_pred - eps)**2
                total_loss += loss

                # backprop through MLP
                grad = 2*(eps_pred - eps)
                self.mlp.backward(grad)

                # SGD step
                for layer in (self.mlp.l1, self.mlp.l2):
                    layer.weight -= lr * layer.dW
                    layer.bias   -= lr * layer.db

            avg_loss = total_loss / n
            if ep % 10 == 0 or ep==1:
                print(f"Epoch {ep:4d}/{epochs} — loss: {avg_loss:.6f}")


class Diffusion:
    def __init__(self, beta=0.1):
        self.mlp   = MLP()
        self.beta  = beta
        self.gamma = 1 - beta

    def forward_diffuse(self, x0, t):
        gamma_t = self.gamma**t
        eps     = np.random.randn(*np.shape(x0))
        x_t     = np.sqrt(gamma_t)*x0 + np.sqrt(1-gamma_t)*eps
        return x_t, eps, gamma_t

    def backward_step(self, x_t, t):
        # Build an (N,2) input: [x_t, t]
        xt_col   = np.atleast_1d(x_t).reshape(-1,1)          # (N,1)
        t_col    = np.full_like(xt_col, fill_value=t, dtype=float)
        mlp_in   = np.hstack([xt_col, t_col])                # (N,2)

        # Predict noise
        eps_pred = self.mlp.forward(mlp_in).reshape(-1)       # (N,)

        gamma_t  = self.gamma**t
        beta_t   = self.beta

        # Posterior mean
        mean = (1/np.sqrt(1-beta_t)) * (
            x_t - (beta_t/np.sqrt(1-gamma_t)) * eps_pred
        )

        # Add noise for t>1
        if t > 1:
            noise = np.random.randn(*x_t.shape)
        else:
            noise = 0

        return mean + np.sqrt(beta_t)*noise

    def sample(self, T, shape):
        # Start from Gaussian noise
        x = np.random.randn(*shape)
        for t in range(T, 0, -1): #go backward t times
            x = self.backward_step(x, t)
        return x
    
    def train(self, data, epochs=100, T=50, lr=1e-3):
        n = data.shape[0]
        for ep in range(1, epochs+1):
            perm, total_loss = np.random.permutation(n), 0.0
            for i in perm:
                x0 = data[i]
                # pick t, diffuse
                t = np.random.randint(1, T+1)
                x_t, eps, _ = self.forward_diffuse(x0, t)

                # predict and loss
                inp      = np.array([x_t, t])
                eps_pred = self.mlp.forward(inp)
                loss     = (eps_pred - eps)**2
                total_loss += loss

                # backprop through MLP
                grad = 2*(eps_pred - eps)
                self.mlp.backward(grad)

                # SGD step
                for layer in (self.mlp.l1, self.mlp.l2):
                    layer.weight -= lr * layer.dW
                    layer.bias   -= lr * layer.db

            avg_loss = total_loss / n
#            if ep % 10 == 0 or ep==1:
            print(f"Epoch {ep}/{epochs} — loss: {avg_loss}")
