import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random

'''
    Create a program to calculate the coefficients of the multilayer perceptron. 
    The multilayer perceptron must perform the function of an approximator. 
    Structure of the multilayer perceptron:

    one input (input 20 input vectors (20 examples) X, with values in the range 0 to 1, eg x = 0.1: 1/22: 1;).
    one output (for example, 
    the output is expecting the desired response that 
    can be calculated using the formula: y = (1 + 0.6 * sin (2 * pi * x / 0.7)) + 0.3 * sin (2 * pi * x)) / 2; -
      the neural network being created should "model / simulate the behavior of this formula" using a completely different
        mathematical expression than this);


    One hidden layer with hyperbolic tangent or sigmoidal activation functions
      in neurons (number of neurons: 4-8);
    linear activation function in the output neuron;
    training algorithm - Backpropagation.
'''

learning_rate = 0.15

# 20 Input vectors, range from 0.1 to 1
X = np.arange(0.1, 1.01, 1/22)

d = ((1 + 0.6 * np.sin(2 * np.pi * X / 0.7)) + 0.3 * np.sin(2 * np.pi * X)) / 2

plt.figure(figsize=(10, 6))
plt.plot(X, d, 'kx', label='Desired Output')
# plt.show()

# Initialize Weights and Biases
w11_1 = np.random.randn()
w21_1 = np.random.randn()
b1_1 = np.random.randn()
w11_2 = np.random.randn()
w12_2 = np.random.randn()
b2_1 = np.random.randn()
b1_2 = np.random.randn()

# Sigmoid activation function
def YFunction(v):
    return 1 / (1 + np.exp(-v))

# Derivative of Sigmoid
def sigmoid_derivative(y):
    return y * (1 - y)

Y = [0] * len(X)
total_error = 0

# Training Loop
epochs = 10000
for t in range(1, epochs):
    for i in range(len(X)):
        # Forward Pass
        v1 = X[i] * w11_1 + b1_1
        v2 = X[i] * w21_1 + b2_1

        # Activation function
        y1_1 = YFunction(v1)
        y2_1 = YFunction(v2)

        # Output layer (Linear activation)
        v1_2 = y1_1 * w11_2 + y2_1 * w12_2 + b1_2

        Y[i] = v1_2
 
        # Training part:
        #### Backpropagation ####
        # Error Calculation
        error = d[i] - Y[i]
        # Estimating error gradient ofr output neuron
        delta_output = error

        # Calculate deltas for hidden neurons
        delta1 = sigmoid_derivative(y1_1) * delta_output * w11_1
        delta2 = sigmoid_derivative(y2_1) * delta_output * w12_2

        # Update Weights and Biases
        # Hidden layer weights and biases
        w11_1 += learning_rate * delta1 * X[i]
        w21_1 += learning_rate * delta2 * X[i]
        b1_1 += learning_rate * delta1
        b2_1 += learning_rate * delta2

        # Output layer weights and bias
        w11_2 += learning_rate * delta_output * y1_1
        w12_2 += learning_rate * delta_output * y2_1
        b1_2 += learning_rate * delta_output


# After Training Calculation
#outputs = []  # Reset outputs list for final predictions
for i in range(len(X)):
        # Forward Pass
        v1 = X[i] * w11_1 + b1_1
        v2 = X[i] * w21_1 + b2_1

        # Activation function
        y1_1 = YFunction(v1)
        y2_1 = YFunction(v2)

        # Output layer (Linear activation)
        v1_2 = y1_1 * w11_2 + y2_1 * w12_2 + b1_2

        Y[i] = v1_2

# Plot MLP Outputs
plt.plot(X, Y, 'r-', label='Trained')
plt.xlabel('Input X')
plt.ylabel('Output Y')
plt.legend()
plt.grid(True)
plt.show()