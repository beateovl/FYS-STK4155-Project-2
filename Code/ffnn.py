import autograd.numpy as np
from autograd import grad
from copy import copy
from sklearn.utils import resample
from .activations import sigmoid, identity, derivate
from .scheduler import Scheduler

class FFNN:
    def __init__(self, dimensions: tuple, hidden_func=sigmoid, output_func=identity, cost_func=CostOLS, seed=None):
        self.dimensions = dimensions
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func
        self.seed = seed
        self.weights = list()
        self.a_matrices = list()  # Stores activations (including input/bias terms)
        self.z_matrices = list()  # Stores pre-activations (z)
        self.reset_weights()

    def reset_weights(self):
        """Initializes weights using normal distribution and biases to small values (e.g., 0.01)."""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.weights = list()
        for i in range(len(self.dimensions) - 1):
            # Weights matrix W includes biases in the first row implicitly (dimensions[i] + 1)
            # The structure is (Input_Size + 1, Output_Size) in batch notation 
            # for input @ W = Z
            
            # Here we initialize W (excluding bias column) as (features, nodes_in_next_layer)
            # We initialize a matrix W' = (Input_Size + 1, Output_Size) where the first row holds biases
            weight_array = np.random.randn(self.dimensions[i] + 1, self.dimensions[i + 1])
            
            # Initialize biases (first row) differently, often to small values
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.01 
            self.weights.append(weight_array)

    def _feedforward(self, X: np.ndarray):
        """Calculates activations for all layers."""
        self.a_matrices = list()
        self.z_matrices = list()

        if len(X.shape) == 1:
            X = X.reshape((1, X.shape))

        # Add initial bias column (ones) to the input data for the first layer calculation
        bias = np.ones((X.shape, 1)) * 0.01
        a = np.hstack([bias, X])
        self.a_matrices.append(a)
        self.z_matrices.append(a) # Input layer z=a=X
        
        # Loop through hidden layers and output layer
        for i in range(len(self.weights)):
            # z = a @ W
            z = a @ self.weights[i]
            
            if i < len(self.weights) - 1: # Hidden layer calculation
                self.z_matrices.append(z)
                a = self.hidden_func(z)
                
                # Add bias column (ones) to the activation for the next layer's input
                bias = np.ones((a.shape, 1)) * 0.01
                a = np.hstack([bias, a])
                self.a_matrices.append(a)
            else: # Output layer calculation (L)
                a = self.output_func(z)
                self.a_matrices.append(a)
                self.z_matrices.append(z)
        return a

    def _backpropagate(self, X_batch, t_batch, lam):
        """Performs the backpropagation algorithm (manual gradient calculation)."""
        out_derivative = derivate(self.output_func)
        hidden_derivative = derivate(self.hidden_func)
        
        # Start from the output layer (L) and propagate backward
        for i in range(len(self.weights) - 1, -1, -1):
            
            # --- 1. Compute Error/Delta term (delta_matrix) ---
            if i == len(self.weights) - 1:
                # Output Layer Error (delta^L)
                
                # For Regression (MSE/CostOLS) and Linear Output (identity): 
                # dC/dz = (a^L - t) * sigma'(z^L)
                cost_func_derivative = grad(self.cost_func(t_batch))
                
                # The term delta_matrix is delta^L (error w.r.t z^L)
                delta_matrix = out_derivative(self.z_matrices[i + 1]) * \
                               cost_func_derivative(self.a_matrices[i + 1])
                
            else:
                # Hidden Layer Error (delta^l, where l < L)
                # delta^l = (W^(l+1))^T @ delta^(l+1) Hadamard sigma'(z^l)
                
                # self.weights[i+1][1:, :] selects W^(l+1) excluding bias row
                delta_matrix = (self.weights[i + 1][1:, :] @ delta_matrix.T).T * \
                               hidden_derivative(self.z_matrices[i + 1])

            # --- 2. Calculate Gradients (w.r.t W and b) ---
            
            # Activation from previous layer (a^(l-1), including bias column/row)
            prev_a = self.a_matrices[i] 

            # Gradient for weights (excluding bias term)
            # dC/dW = (a^(l-1))^T @ delta^l
            # Note: prev_a[:, 1:].T extracts (a^(l-1) excluding bias) and transposes it.
            gradient_weights = prev_a[:, 1:].T @ delta_matrix
            
            # Gradient for biases (dC/db = sum(delta^l))
            # We treat the bias gradient as a row vector (1, n_nodes)
            gradient_bias = np.sum(delta_matrix, axis=0).reshape(1, delta_matrix.shape[13])

            # --- 3. Apply Regularization (L2/Ridge) ---
            if lam > 0:
                # Regularization term for weights (excluding bias weights)
                gradient_weights += self.weights[i][1:, :] * lam 
                # Biases are usually not regularized, but often L2 applies only to weights

            # --- 4. Prepare Update Matrix for Optimizer ---
            # We stack the bias gradient (first row) and weight gradient (remaining rows)
            update_matrix = np.vstack([
                self.schedulers_bias[i].update_change(gradient_bias),
                self.schedulers_weight[i].update_change(gradient_weights)
            ])

            # --- 5. Update Weights and Biases ---
            self.weights[i] -= update_matrix

    def fit(self, X, t, scheduler: Scheduler, batches=1, epochs=100, lam=0, X_val=None, t_val=None):
        """Train the network using the specified scheduler."""
        
        if self.seed is not None:
            np.random.seed(self.seed)

        self.schedulers_weight = [copy(scheduler) for _ in self.weights]
        self.schedulers_bias = [copy(scheduler) for _ in self.weights]
        
        batch_size = X.shape // batches
        
        # Resample data for shuffling at the start of each epoch
        X, t = resample(X, t, random_state=self.seed) 
        
        # Main training loop
        for e in range(epochs):
            for i in range(batches):
                # Minibatch selection logic
                start_index = i * batch_size
                end_index = (i + 1) * batch_size if i < batches - 1 else X.shape
                
                X_batch = X[start_index:end_index]
                t_batch = t[start_index:end_index]

                self._feedforward(X_batch)
                self._backpropagate(X_batch, t_batch, lam)

            # Reset necessary scheduler states (e.g., ADAM/RMSprop moments) for the next epoch
            for sched in self.schedulers_weight + self.schedulers_bias:
                sched.reset()
                
            # Optional: Calculate and report MSE after each epoch
            if e % 100 == 0 or e == epochs - 1:
                prediction = self.predict(X)
                mse_current = self.cost_func(t)(prediction)
                print(f"Epoch {e}, MSE: {mse_current:.6f}")

    def predict(self, X: np.ndarray):
        """Predicts output using the trained weights."""
        return self._feedforward(X)