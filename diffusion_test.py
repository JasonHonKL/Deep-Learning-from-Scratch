import numpy as np
import matplotlib.pyplot as plt
from diffusion import Diffusion

# assume MLP, Linear, Sigmoid and Diffusion are defined/imported above
# from your_module import Diffusion

def main():
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Data ~ N(0,1)
    data = np.random.normal(0 ,10 ,10000)

    # 2) Train
    diff = Diffusion(beta=0.1)
    diff.train(data, epochs=10, T=30, lr=1e-3)

    # 3) Sample
    samples = diff.sample(T=30, shape=(10000,))

    # 4) Plot
    plt.hist(data,    bins=100, density=True, alpha=0.5, label='True')
    plt.hist(samples, bins=100, density=True, alpha=0.5, label='Generated')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()