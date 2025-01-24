# ASSIGNMENT 4 EXERCISE 3
import numpy as np
import matplotlib.pyplot as plt

L = 50000  # number of samples
W = 3
# channel impulse response
h = np.array([0.5 * (1 + np.cos(2 * np.pi * (k - 2) / W)) if k in [1, 2, 3] else 0 for k in range(0,5)])
#sigma_n = 10e-3
n = 0#np.random.normal(0, sigma_n, L) # Gaussian noise n
a = 0.8

# Exercise 3.3.1
# Generate the data sequence x[k].
x_train = np.random.choice([1, -1], size=L) # random sequence of +1 and -1
# Plot the data sequence
plt.figure(figsize=(10, 4))
plt.stem(x_train[:100])  # plotting only the first 100
plt.title('Data generation')
plt.xlabel('Index')
plt.ylabel('a.u.')
plt.grid(True)
plt.show()

# Plot the data sequence after it has passed the nonlinear channel.
# Obtain y[k]
convolved = np.convolve(x_train, h, mode='full')[:L]
Y_train = (convolved + a * (convolved ** 2) + n)

# Plot the output of the channel
plt.figure(figsize=(10, 4))
plt.stem(Y_train[:200]) 
plt.title('Channel output y[k]')
plt.xlabel('Index')
plt.ylabel('a.u.')
plt.grid(True)
plt.show()

# Exercise 3.3.2
# Implement the nonlinear adaptive equalizer
M = 11
D = 7
n_nodes = 100
sigma2 = 0.09
mu = 0.01  # Learning rate for gradient descent
w = np.zeros(M)
y_hat = np.zeros(L)
y_pred_final = np.zeros(L)
errors = np.zeros(L)

for epoch in range(200): 
    # Run the nonlinear equalizer several times, each time with different weights initialization seed
    W1 = np.random.normal(0, np.sqrt(sigma2), (M + 1, n_nodes))
    W2 = np.random.normal(0, np.sqrt(sigma2), (n_nodes + 1, 1))
    # We feed the NN with the output of the channel Y_train
    # The goal is to perform equalization and get back
    # An estimation of the original input signal x_train
    for k in range(M + D, L): # Implementing slide 32 update algorithm
        # Add bias column to Y_train
        y_window = np.append(Y_train[:, np.newaxis][k - M:k], 1) # Select M recent samples from Y_train (M + 1, 1)
        
        # Forward pass to hidden layer
        A = y_window.T @ W1 # (1, n_nodes)    
        
        # Apply activation function np.tanh() and add bias column
        Z = np.append(np.tanh(A), 1) # (n_nodes + 1, 1)
        
        # Compute predictions from output layer
        y_hat[k] = Z @ W2 # (L, )
        
        # Backward pass: Compute the gradient and update the weights
        # Gradient of the error with respect to W2
        dE_dW2 = Z * (y_hat[k] - x_train[k]) # (1, n_nodes+1)
        W2 -= mu * dE_dW2[:, np.newaxis]  # Update rule for W2 (n_nodes+1, 1)
        
        # Gradient of the error with respect to W1
        dE_dW1 = (y_hat[k] - x_train[k]) * (1 - np.tanh(A)**2).T[:, np.newaxis] * W2[:-1] * y_window # (n_nodes, n_nodes + 2)
        W1 -= mu * dE_dW1.T # (2, n_nodes)
        
        errors[k] = y_hat[k] - x_train[k - D] # Error between delayed input and predicted output

# After training, use the final updated weights to predict. TESTING
x_test = np.random.choice([1, -1], size=L) # (L,1)
convolved = np.convolve(x_test, h, mode='full')[:L]
Y_test = (convolved + a * (convolved ** 2) + n)
for k in range(M + D, L): # Implementing slide 32 update algorithm
    # Add bias column to Y_test
    y_window = np.append(Y_test[:, np.newaxis][k - M:k], 1) # Select M recent samples from Y_est (M + 1, 1)
    
    # Forward pass to hidden layer
    A = y_window.T @ W1 # (1, n_nodes)    
    
    # Apply activation function np.tanh() and add bias column
    Z = np.append(np.tanh(A), 1) # (n_nodes + 1, 1)
    
    # Compute predictions from output layer
    y_hat[k] = Z @ W2 # (L, )
    
    # Backward pass: Compute the gradient and update the weights
    # Gradient of the error with respect to W2
    dE_dW2 = Z * (y_hat[k] - x_test[k]) # (1, n_nodes+1)
    W2 -= mu * dE_dW2[:, np.newaxis]  # Update rule for W2 (n_nodes+1, 1)
    y_pred_final[k] = Z @ W2 # (L,)
    
    # Gradient of the error with respect to W1
    dE_dW1 = (y_hat[k] - x_test[k]) * (1 - np.tanh(A)**2).T[:, np.newaxis] * W2[:-1] * y_window # (n_nodes, n_nodes + 2)
    W1 -= mu * dE_dW1.T # (2, n_nodes)
    
    #errors[k] = y_hat[k] - Y_train[k - D] # Error between delayed input and predicted output

# Plot the original input and the equalized output
plt.figure(figsize=(15, 6))

# Plot original input signal
plt.subplot(2, 1, 1)  # two rows, one column, first subplot
plt.stem(x_test[:200], linefmt='C0-', markerfmt='C0o')
plt.title('Original input signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

y_pred_final = np.sign(y_pred_final)
# Plot equalized output signal
plt.subplot(2, 1, 2)  # two rows, one column, second subplot
plt.stem(y_pred_final[:200], linefmt='C1-', markerfmt='C1o')
plt.title('Equalized Output Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Compute the number or errors between the equalized signal and the original data sequence  x[k]
cross_corr = np.correlate(x_test, y_pred_final, mode='full')
# Find the index of the maximum correlation value
delay = np.argmax(cross_corr) - (len(x_test) - 1)

print(f'Detected delay: {-delay}')

# Calculate the error signal
error_signal = x_test[:len(x_test)+delay] - y_pred_final[-delay:]

# Count the number of mismatches (errors)
#number_of_errors = np.sum(y_pred_final != x_test)
number_of_errors = np.sum(np.abs(error_signal) >= 0.1)

# Calculate the error rate
error_rate = number_of_errors / len(error_signal)

print(f"Number of Errors: {number_of_errors}")
print(f"Error Rate: {error_rate * 100:.2f}%")
