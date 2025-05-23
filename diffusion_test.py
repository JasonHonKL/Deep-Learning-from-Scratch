from diffusion import MLP
import numpy as np

np.random.seed(42)
N = 2000  # number of samples

# Inputs: x and z (shape: N x 2)
x = np.random.uniform(-2, 2, size=(N, 1))
z = np.random.uniform(-2, 2, size=(N, 1))
X = np.hstack([x, z])  # shape (N, 2)

# True output with noise
noise = 0.1 * np.random.randn(N, 1)
y_true = (x ** 2) + z + noise  # shape (N, 1)

# Initialize model
model = MLP()

# Training loop
epochs = 100

for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(X)  # shape (N, 1)

    # Compute mean squared error loss and gradient
    loss = np.mean((y_pred - y_true)**2)
    grad_loss = 2 * (y_pred - y_true) / N  # gradient of MSE loss w.r.t predictions

    # Backward pass
    model.backward(grad_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d} / {epochs}, Loss: {loss:.6f}")

# Final evaluation
print("\nTraining complete.")
print("Sample predictions vs true values:")
for i in range(5):
    print(f"Pred: {y_pred[i,0]:.4f}, True: {y_true[i,0]:.4f}")