import model
import numpy as np

'''
Example Usage
    a = cnn.Linear(1 ,3)
    s = cnn.Sigmoid()
    l = cnn.ANN()
    pred = l(np.array([
        [1,2],
        [2,2],
        [3,5],
    ]))
'''

## Assume the line is y = x^2 + 2z (which should be learn though the network)

## x and z will be feed into the ANN network 
## Noise is unknown to the network 
x = np.random.randint(1, 12, size=100)
z = np.random.randint(1, 12, size=100)
i = np.column_stack((x, z))  # Shape (10, 2)
noise = np.random.normal(loc=0, scale=0.1, size=(100, 1))  # Shape (10, 1)
y = (x**2 + 2*z + noise.T).T  # Shape (10, 1)

# Initialize the neural network
nn = model.ANN()

# Training loop
for epoch in range(100):
    print(f"Epoch {epoch+1}: ... ")
    nn.zero_grad()  # Reset gradients
    pred = nn(i)  # Use ANN.__call__ to process the entire batch (i has shape (10, 2))
    loss_fn = model.MSELoss()
    loss = loss_fn(pred, y)  # Compute loss
    nn.backward(loss_fn)  # Backpropagate
    print(f"Loss: {np.mean(loss)}")  # Print average loss for the batch

# Final predictions
pred = nn(i)  # Get predictions for the entire batch
print("Predictions:\n", pred)
print("True values:\n", y)

print(nn(np.array([10 , 10])))