import autograd.numpy as np

def ReLU(z):
    return np.where(z > 0, z, 0) # Example activation function [1, 2]

# Leaky RELU
def LRELU(z):
    delta = 10e-4
    return np.where(z > np.zeros(z.shape), z, delta * z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z)) # Example activation function [1, 2]

def ReLU_der(z):
    return np.where(z > 0, 1, 0) # Example activation derivative [2]

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z)) # Example sigmoid derivative [6]

def mse(predict, target):
    return np.mean((predict - target) ** 2) # Example MSE cost function [5]

def mse_der(predict, target):
    n = predict.shape
    return (2 / n) * (predict - target) # Example derivative of MSE [6]

def cross_entropy(predict, target):
    # Simplified Cross-Entropy for classification [7]
    return np.sum(-target * np.log(predict))