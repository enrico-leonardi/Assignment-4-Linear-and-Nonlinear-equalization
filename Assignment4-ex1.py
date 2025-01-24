# ASSIGNMENT 4 EXERCISE 1
import numpy as np
import matplotlib.pyplot as plt

# Exercise 3.1.1
# Implement linear discrete-time communication system model

L = 10000  # number of samples
x = np.random.choice([1, -1], size=L) # random sequence of +1 and -1

# Plot the data sequence
plt.figure(figsize=(10, 4))
plt.stem(x[:100])  # plotting only the first 100
plt.title('Data generation')
plt.xlabel('Index k')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Exercise 3.1.2
# Use the convolution to obtain the output of the channel
W = 3
sigma_n = 10e-3

# channel impulse response
h = np.array([0.5 * (1 + np.cos(2 * np.pi * (k - 2) / W)) if k in [1, 2, 3] else 0 for k in range(0,5)])
n = np.random.normal(0, sigma_n, L) # Gaussian noise n
# Convolve the data sequence x[k] with the channel impulse response h[k]
y = np.convolve(x, h, mode='full')[:L] + n

# Plot the output of the channel
plt.figure(figsize=(10, 4))
plt.stem(y[:200]) 
plt.title('Channel output y[k]')
plt.xlabel('Index k')
plt.ylabel('Channel output')
plt.grid(True)
plt.show()

# Exercise 3.1.3
# Implement the linear adaptive equalizer 

# Parameters
M = 11 # Number of taps in the equalizer
mu = 0.075  # Learning rate for gradient descent
D = 7 # Delay in samples to align the equalizer output with the desired signal
w = np.zeros(M)
errors = np.zeros(L)

# Output initialization
x_hat = np.zeros(L)

# The input signal 𝑥[𝑘] should be delayed by 𝐷
# samples because the channel and equalizer
# will take some time to respond. The equalizer
# should predict 𝑥^[𝑘] based on a delayed version of 
# 𝑥[𝑘] and the current weights.
# This implies that you need to compute the error
# signal between the predicted output 𝑥^[𝑘] (from
# the equalizer) and the actual signal delayed by 𝐷 samples.

for k in range(M + D, L): # Implementing slide 32 update algorithm
    y_window = y[k - M:k]  # Select M recent samples from y
    x_hat[k] = w.T @ y_window
    errors[k] = x[k - D] - x_hat[k] # Error between delayed input and predicted output
    w += mu * errors[k] * y_window

# Convolve the channel output with the equalizer weights
eq_output = np.convolve(y, w, mode='full')[:L]

# Plot the equalizer output
plt.figure(figsize=(10, 4))
plt.stem(eq_output[:200])
plt.title('Equalizer Output (First 200 Samples)')
plt.xlabel('Index k')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Exercise 3.1.4
# Plot the error squared as a function of number of iterations.

# Plot the squared errors
plt.figure(figsize=(10, 4))
plt.plot(errors[:500] ** 2)
plt.title('Squared Error over Iterations')
plt.xlabel('Iteration k')
plt.ylabel('Squared Error')
plt.grid(True)
plt.show()

# Exercise 3.1.5
# Once the equalizer has converged and the weights of the equalizer has been learned,
# show that the distorted signal after the channel output can be equalized

# Plot the original input and the equalized output
plt.figure(figsize=(15, 6))

# Plot original input signal
plt.subplot(2, 1, 1)  # two rows, one column, first subplot
plt.stem(x[:200], linefmt='C0-', markerfmt='C0o')
plt.title('Original Input Signal x[k]')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

# Plot equalized output signal
plt.subplot(2, 1, 2)  # two rows, one column, second subplot
plt.stem(eq_output[:200], linefmt='C1-', markerfmt='C1o')
plt.title('Equalized Output Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Exercise 3.1.6
# Compute the number or errors between the equalized signal and the original data sequence  x[k]
cross_corr = np.correlate(x, eq_output, mode='full')
# Find the index of the maximum correlation value
delay = np.argmax(cross_corr) - (len(x) - 1)

print(f'Detected delay: {-delay}')

# Calculate the error signal
error_signal = x[:len(x)+delay] - eq_output[-delay:]

# Count the number of mismatches (errors)
number_of_errors = np.sum(np.abs(error_signal) >= 0.1)

# Calculate the error rate
error_rate = number_of_errors / len(error_signal)

print(f"Number of Errors: {number_of_errors}")
print(f"Error Rate: {error_rate * 100:.2f}%")

