from nn.linear import Linear:

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