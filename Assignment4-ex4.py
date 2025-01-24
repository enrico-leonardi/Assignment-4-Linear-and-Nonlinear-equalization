# ASSIGNMENT 4 EXERCISE 4
import numpy as np
import matplotlib.pyplot as plt

# Implement linear discrete-time communication system model
L = 50000  # number of samples
x = np.random.choice([1, -1], size=L) # random sequence of +1 and -1

# Plot the data sequence
plt.figure(figsize=(10, 4))
plt.stem(x[:100])  # plotting only the first 100
plt.title('Data generation')
plt.xlabel('Index k')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Use the convolution to obtain the output of the channel
W = 3
a = 0.8
sigma_n = 10e-3

# channel impulse response
h = np.array([0.5 * (1 + np.cos(2 * np.pi * (k - 2) / W)) if k in [1, 2, 3] else 0 for k in range(0,5)])
n = np.random.normal(0, sigma_n, L) # Gaussian noise n
# Convolve the data sequence x[k] with the channel impulse response h[k]
convolved = np.convolve(x, h, mode='full')[:L]
y = (convolved + a * (convolved ** 2) + n)

# Plot the output of the channel
plt.figure(figsize=(10, 4))
plt.stem(y[:200]) 
plt.title('Channel output y[k]')
plt.xlabel('Index k')
plt.ylabel('Channel output')
plt.grid(True)
plt.show()

# ---------------- Implement the linear adaptive equalizer 
# Parameters
M = 11 # Number of taps in the equalizer
mu = 0.033  # Learning rate for gradient descent
D = 7 # Delay in samples to align the equalizer output with the desired signal
w = np.zeros(M)
errors = np.zeros(L)

# Output initialization
x_hat = np.zeros(L)

for k in range(M + D, L): # Implementing slide 32 update algorithm
    y_window = y[k - M:k]  # Select M recent samples from y
    x_hat[k] = w.T @ y_window
    errors[k] = x[k - D] - x_hat[k] # Error between delayed input and predicted output
    w += mu * errors[k] * y_window

# Plot the error squared as a function of number of iterations.
plt.figure(figsize=(10, 4))
plt.plot(errors[:500] ** 2)
plt.title('Squared Error over Iterations')
plt.xlabel('Iteration k')
plt.ylabel('Squared Error')
plt.grid(True)
plt.show()

# Convolve the channel output with the equalizer weights
eq_output = np.convolve(y, w, mode='full')[:L]
eq_output = np.sign(eq_output)

# Plot the equalizer output
plt.figure(figsize=(10, 4))
plt.stem(eq_output[:200])
plt.title('Equalizer Output (First 200 Samples)')
plt.xlabel('Index k')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Implement the nonlinear adaptive equalizer
#M = 11
#D = 7
n_nodes = 100
sigma2 = 0.09
mu = 0.01  # Learning rate for gradient descent
y_hat = np.zeros(L)
y_pred_final = np.zeros(L)
#errors = np.zeros(L)

for epoch in range(300): 
    # Run the nonlinear equalizer several times, each time with different weights initialization seed
    W1 = np.random.normal(0, np.sqrt(sigma2), (M + 1, n_nodes))
    W2 = np.random.normal(0, np.sqrt(sigma2), (n_nodes + 1, 1))
    # We feed the NN with the output of the channel Y_train
    # The goal is to perform equalization and get back
    # An estimation of the original input signal x_train
    for k in range(M + D, L): # Implementing slide 32 update algorithm
    
        y_window = np.append(eq_output[:, np.newaxis][k - M:k], 1) # Select M recent samples from Y_train (M + 1, 1)
        
        A = y_window.T @ W1 # (1, n_nodes)    
        
        Z = np.append(np.tanh(A), 1) # (n_nodes + 1, 1)
        
        y_hat[k] = Z @ W2 # (L, )
        
        dE_dW2 = Z * (y_hat[k] - x[k]) # (1, n_nodes+1)
        W2 -= mu * dE_dW2[:, np.newaxis]  # Update rule for W2 (n_nodes+1, 1)
        
        dE_dW1 = (y_hat[k] - x[k]) * (1 - np.tanh(A)**2).T[:, np.newaxis] * W2[:-1] * y_window # (n_nodes, n_nodes + 2)
        W1 -= mu * dE_dW1.T # (2, n_nodes)
        
        errors[k] = y_hat[k] - x[k - D] # Error between delayed input and predicted output


# Plot the error squared as a function of number of iterations.
plt.figure(figsize=(10, 4))
plt.plot(errors[:500] ** 2)
plt.title('Squared Error over Iterations')
plt.xlabel('Iteration k')
plt.ylabel('Squared Error')
plt.grid(True)
plt.show()

# After training, use the final updated weights to predict ------------ TESTING
x_test = np.random.choice([1, -1], size=L) # (L,1)
convolved = np.convolve(x_test, h, mode='full')[:L]
y_test = (convolved + a * (convolved ** 2) + n)

# Convolve the channel output with the equalizer weights
eq_output = np.convolve(y_test, w, mode='full')[:L]
eq_output = np.sign(eq_output)

for k in range(M + D, L): # Implementing slide 32 update algorithm
    # Add bias column to Y_test
    y_window = np.append(eq_output[:, np.newaxis][k - M:k], 1) # Select M recent samples from Y_est (M + 1, 1)
    
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

# Once the equalizer has converged and the weights of the equalizer has been learned,
# show that the distorted signal after the channel output can be equalized
plt.figure(figsize=(15, 6))

# Plot original input signal
plt.subplot(2, 1, 1)  # two rows, one column, first subplot
plt.stem(x_test[:200], linefmt='C0-', markerfmt='C0o')
plt.title('Original Input Signal x[k]')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

# Plot equalized output signal
plt.subplot(2, 1, 2)  # two rows, one column, second subplot
plt.stem(eq_output[:200], linefmt='C1-', markerfmt='C1o')
plt.title('Linearly Equalized Output Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Compute the number or errors between the equalized signal and the original data sequence  x[k]
cross_corr = np.correlate(x_test, eq_output, mode='full')
# Find the index of the maximum correlation value
delay = np.argmax(cross_corr) - (len(x_test) - 1)

print(f'Detected delay: {-delay}')

# Calculate the error signal
error_signal = x_test[:len(x)+delay] - eq_output[-delay:]

# Count the number of mismatches (errors)
number_of_errors = np.sum(np.abs(error_signal) >= 0.1)

# Calculate the error rate
error_rate = number_of_errors / len(error_signal)

print(f"Number of Errors: {number_of_errors}")
print(f"Error Rate: {error_rate * 100:.2f}%")

# Plot the original input and the equalized output
plt.figure(figsize=(15, 6))

# Plot original input signal
plt.subplot(2, 1, 1)  # two rows, one column, first subplot
plt.stem(x_test[:200], linefmt='C0-', markerfmt='C0o')
plt.title('Original input signal x[k]')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

y_pred_final = np.sign(y_pred_final)
# Plot equalized output signal
plt.subplot(2, 1, 2)  # two rows, one column, second subplot
plt.stem(y_pred_final[:200], linefmt='C1-', markerfmt='C1o')
plt.title('Nonlinearly Equalized Output Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Compute the number or errors between the equalized signal and the original data sequence x[k]
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
