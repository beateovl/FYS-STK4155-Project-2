import autograd.numpy as np

# --- 1. Sigmoid Function and Derivative ---

def sigmoid(z):
    """
    The logistic sigmoid activation function, defined as 1 / (1 + e^-z).
    This function is commonly used in neural networks.
    """
    # Implemented using the standard logistic function definition
    return 1.0 / (1.0 + np.exp(-z)) 

def dsigmoid(z):
    """
    The derivative of the sigmoid function, calculated as a * (1 - a),
    where a = sigmoid(z).
    """
    a = sigmoid(z)
    # The derivative expression is a(1 - a)
    return a * (1 - a)

# --- 2. ReLU Function and Derivative ---

def relu(z):
    """
    The Rectified Linear Unit (ReLU) activation function, defined as max(0, z).
    ReLU does not saturate for positive values, alleviating the vanishing 
    gradient problem.
    """
    # Uses np.where to return z if z > 0, else 0
    return np.where(z > 0, z, 0) 

def drelu(z):
    """
    The derivative of the ReLU function, which is 1 if z > 0, and 0 otherwise.
    """
    # Returns 1 if z > 0, and 0 otherwise
    return np.where(z > 0, 1, 0) 

# --- 3. Leaky ReLU Function and Derivative ---

def leaky_relu(z, a=0.01):
    """
    The Leaky ReLU (LRELU) activation function [13].
    It uses a small slope 'a' for negative inputs to address the 'dying ReLU' problem.
    """
    # If z > 0, returns z; otherwise, returns a * z.
    return np.where(z > np.zeros(z.shape), z, a * z)

def dleaky_relu(z, a=0.01):
    """
    The derivative of the Leaky ReLU function, which is 1 for positive inputs 
    and the slope 'a' for non-positive inputs. 
    """
    # If z > 0, returns 1; otherwise, returns 'a'.
    return np.where(z > 0, 1, a)