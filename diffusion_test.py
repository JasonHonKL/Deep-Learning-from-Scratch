from diffusion import Diffusion
import numpy as np


diff = Diffusion()
x = np.array([1.0, 2.0, 3.0])  # example input
t = 1000  # timestep

x_t_loop, gamma_t_loop = diff.forward(x, t)

# Calculate gamma_t using power function directly
gamma_t_power = diff.gamma ** t

print(f"Gamma_t from loop: {gamma_t_loop}")
print(f"Gamma_t from power: {gamma_t_power}")
print(f"Difference: {abs(gamma_t_loop - gamma_t_power)}")

# Check if gamma_t_loop and gamma_t_power are almost equal
assert np.isclose(gamma_t_loop, gamma_t_power), "Gamma_t values do not match!"

print(f"x_t: {x_t_loop}")