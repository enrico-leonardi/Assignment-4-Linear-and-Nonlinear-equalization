# ASSIGNMENT 4 EXERCISE 2
import numpy as np
import matplotlib.pyplot as plt

# Initialize
sigma2 = 0.09
N = 300
n_nodes = 10
mu = 0.01  # Learning rate for gradient descent

x_train = np.random.uniform(-1, 1, N)

# Define the functions
y1 = x_train ** 2
y2 = x_train ** 3
y3 = np.sin(x_train)
y4 = np.abs(x_train)

# Stack them into a single 2D array
Y = np.column_stack((y1, y2, y3, y4)) # each column is a different function

mse_history = []

for i in range(Y.shape[1]):
    Y_train = Y[:, i]
    Y_train = Y_train[:, np.newaxis]
    # Add bias column to X_train
    X_train = np.column_stack((x_train, np.ones((len(Y_train), 1))))
    
    # Initialize weights
    W1 = np.random.normal(0, np.sqrt(sigma2), (2, n_nodes))
    W2 = np.random.normal(0, np.sqrt(sigma2), (n_nodes + 1, 1))
    
    # Track the MSE for the current function
    mse_per_iteration = []
    
    for epoch in range(500):
        y_pred = []            
        for k in range(len(X_train[:,0])):
            
            # Forward pass to hidden layer
            A = X_train[k,:] @ W1 # (n_nodes,)
            
            # Apply activation function np.tanh() and add bias column
            Z = np.append(np.tanh(A), 1) # (n_nodes + 1, 1)
        
            # Backward pass: Compute the gradient and update the weights
            # Gradient of the error with respect to W2
            dE_dW2 = Z * (Z @ W2 - Y_train[k,:]) # (n_nodes + 1, 1)
            W2 -= mu * dE_dW2[:, np.newaxis]   # Update rule for W2 (n_nodes + 1, 1)
        
            # Gradient of the error with respect to W1
            dE_dW1 = X_train[k,:] * ((1 - np.tanh(A)**2)[:, np.newaxis] * (Z @ W2 - Y_train[k,:]) * W2[0:-1]) # (n_nodes, 2)
            W1 -= mu * dE_dW1.T # (2, n_nodes)
            
            # Compute predictions from output layer
            y_pred.append(Z @ W2) # (1, )
        # Calculate MSE for this iteration
        mse = np.mean((y_pred - Y_train) ** 2)
        mse_per_iteration.append(mse)

    # Store the MSE for the current function
    mse_history.append(mse_per_iteration)
    
    # After training, use the final updated weights to predict. TESTING
    x_test = np.linspace(-1, 1, N)
    X_test = np.column_stack((x_test, np.ones((len(Y_train), 1))))
    A_final = X_test @ W1
    Z_final = np.column_stack((np.tanh(A_final), np.ones((len(A_final), 1))))
    y_pred_final = Z_final @ W2

    # Plot
    plt.figure()
    plt.scatter(x_train, Y_train, label='Training data')
    plt.scatter(x_test, y_pred_final, label='Predicted', alpha=0.4)
    plt.title(f"Prediction of curve {i}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.show()

# Plot MSE history for each function
plt.figure()
for idx, mse in enumerate(mse_history):
    plt.plot(mse, label=f'Function {idx+1}')
plt.title("Mean Squared Error (MSE) over Iterations")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.grid()
plt.show()