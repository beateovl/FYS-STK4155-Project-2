import autograd.numpy as np
from autograd import grad
from copy import copy
from Code.scheduler import Scheduler

"""
Based on exercises week 42 (exercise 8 structure). 
Implements a feed-forward neural network using batched inputs,
with weights and biases stored separately for clarity.
"""

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
        seed=None,
        l1: float = 0.0,
        l2: float = 0.0
    ):
     
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.seed = seed
        self.l1 = l1
        self.l2 = l2

        last_act = self.activation_funcs[-1].__name__.lower()
        self.classification = (last_act == "softmax")

        if self.seed is not None:
            np.random.seed(self.seed)

        # Create layers (weights and biases)
        self.layers = self._create_layers_batch(network_input_size, layer_output_sizes)

    def _create_layers_batch(self, network_input_size, layer_output_sizes):
        """
        Creates layers based on the batched input approach (W shape: Input size, Output size) [3, 4].
        """
        layers = []
        in_size = self.network_input_size
        for out_size, act in zip(self.layer_output_sizes, self.activation_funcs):
            fan_in, fan_out = in_size, out_size
            name = act.__name__.lower()
            if name in ("relu", "lrelu", "leakyrelu", "leaky_relu"):
                scale = np.sqrt(2.0 / fan_in)          # He init
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier
            W = np.random.randn(fan_in, out_size) * scale
            b = np.zeros(out_size)  # or 0.01 for ReLU/LReLU to reduce dead units
            layers.append((W, b))
            in_size = out_size
        self.layers = layers
        return layers


    def predict(self, inputs):
        """
        Simple feed forward pass for batched inputs, returning the final activation [4, 8].
        """
        a = inputs
        # Iterate over stored layers and activation functions
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            # Z = a @ W + b. a is (N, I), W is (I, O), Z is (N, O)
            z = a @ W + b
            a = activation_func(z)
        return a

    def cost(self, inputs, targets):
        pred = self.predict(inputs)
        base = self.cost_fun(pred, targets)
        if self.l1 != 0.0 or self.l2 != 0.0:
            l1_term = sum(np.sum(np.abs(W)) for W, _ in self.layers)
            l2_term = sum(np.sum(W*W)       for W, _ in self.layers)
            base += self.l1 * l1_term + self.l2 * l2_term
        return base


#storing function for backpropagation
    def _feed_forward_saver(self, inputs):
        """
        Performs the batched feed-forward pass while saving intermediate layer inputs (a) 
        and weighted sums (z) for backpropagation [11, 12].
        """
        layer_inputs = []
        zs = []
        a = inputs
        
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a) # Input activation (a) before current layer [12]
            z = a @ W + b          # Weighted sum [10]
            a = activation_func(z)
            zs.append(z)
        
        return layer_inputs, zs, a # Returns saved data and final prediction [12]

    #backpropagation
    def compute_gradient(self, inputs, targets):
        # Forward pass with caches (your names)
        a_matrices, z_matrices, predict_batch = self._feed_forward_saver(inputs)
        batch_size = inputs.shape[0]
        n_layers = len(self.layers)
        L = n_layers - 1

        layer_grads = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.layers]

        # ----- Output layer delta -----
        if self.classification and self.activation_funcs[L].__name__.lower() == "softmax":
            delta = predict_batch - targets
        else:
            d_act_L = self.activation_ders[L](z_matrices[L])
            dC_dpred = self.cost_der(predict_batch, targets)   # same shape as predict_batch
            delta = dC_dpred * d_act_L

        # Grad for output layer (L)
        dW_L = a_matrices[L].T @ delta
        db_L = np.sum(delta, axis=0)

        # L1/L2 on weights only (not biases)
        W_L, _ = self.layers[L]
        if getattr(self, "l2", 0.0) != 0.0:
            dW_L += self.l2 * W_L
        if getattr(self, "l1", 0.0) != 0.0:
            dW_L += self.l1 * np.sign(W_L)

        layer_grads[L] = (dW_L, db_L)

        # ----- Hidden layers: backprop -----
        for i in range(L - 1, -1, -1):
            W_next, _ = self.layers[i + 1]
            d_act_i = self.activation_ders[i](z_matrices[i])
            delta = (delta @ W_next.T) * d_act_i

            dW_i = a_matrices[i].T @ delta
            db_i = np.sum(delta, axis=0)

            W_i, _ = self.layers[i]
            if getattr(self, "l2", 0.0) != 0.0:
                dW_i += self.l2 * W_i
            if getattr(self, "l1", 0.0) != 0.0:
                dW_i += self.l1 * np.sign(W_i)

            layer_grads[i] = (dW_i, db_i)

        return layer_grads


        

    def update_weights(self, layer_grads, learning_rate):
        """
        Applies gradients directly, simulating a basic Gradient Descent update (P -= eta * Gradient) [16].
        Note: In real use, an external scheduler (like Adam or RMSprop) would manage the update steps.
        """
        for k in range(len(self.layers)):
            W, b = self.layers[k]
            W_g, b_g = layer_grads[k]
            
            # Update rule: P -= learning_rate * Gradient [16]
            W -= learning_rate * W_g
            b -= learning_rate * b_g
            
            self.layers[k] = (W, b)

    def reset_weights(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        pass 


    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        scheduler: Scheduler,       # The optimizer instance (Constant, Adam, etc.)
        batches: int = 1,           # Number of batches (1 for standard GD)
        epochs: int = 100,
        lam: float = 0,
        l1=None,
        l2=None,
    ):
        if l2 is None and lam is not None:
            l2 = float(lam)
        if l1 is not None:
            self.l1 = float(l1)
        if l2 is not None:
            self.l2 = float(l2)                

        N_samples = X.shape[0]

        # Treat `batches` as the requested number of mini-batches per epoch
        batches = max(1, batches)
        batch_size = max(1, int(np.ceil(N_samples / batches)))

        # One scheduler per param (keep your copies)
        schedulers_W = [copy(scheduler) for _ in self.layers]
        schedulers_b = [copy(scheduler) for _ in self.layers]

        train_errors = np.full(epochs, np.nan)
        val_errors   = np.full(epochs, np.nan)

        for epoch in range(epochs):
            # Do NOT reset optimizers each epoch; they need their running moments/history
            # for sch in schedulers_W + schedulers_b: sch.reset()  # <-- remove this

            shuffled_indices = np.random.permutation(N_samples)

            # Single pass over the data in `batches` chunks
            for start_index in range(0, N_samples, batch_size):
                end_index = min(start_index + batch_size, N_samples)
                batch_indices = shuffled_indices[start_index:end_index]

                X_batch = X[batch_indices]
                t_batch = t[batch_indices]

                layer_grads = self.compute_gradient(X_batch, t_batch)

                for k in range(len(self.layers)):
                    W, b = self.layers[k]
                    W_g, b_g = layer_grads[k]
                    W_change = schedulers_W[k].update_change(W_g)
                    b_change = schedulers_b[k].update_change(b_g)
                    W -= W_change
                    b -= b_change
                    self.layers[k] = (W, b)

            # once per epoch
            train_errors[epoch] = self.cost(X, t)

        return {"train_errors": train_errors, "val_errors": val_errors}

    # Methods for Autograd compliance (Optional functionality) [17]

    def autograd_compliant_predict(self, layers, inputs):
        """
        Feed forward pass designed to be differentiable by Autograd by accepting 
        the layers parameter explicitly [18].
        """
        a = inputs
        for (W, b), activation_func in zip(layers, self.activation_funcs):
            z = np.matmul(a, W) + b
            a = activation_func(z)
        return a

    def autograd_gradient(self, inputs, targets):
        """
        Uses autograd.grad to compute the gradients of the cost function w.r.t. the layers [16, 19].
        """
        # Define a cost wrapper function compliant with autograd's expectation 
        # (layers must be one of the function arguments being differentiated) [19]
        def cost_autograd(layers, inputs, targets):
            predict = self.autograd_compliant_predict(layers, inputs)
            return self.cost_fun(predict, targets)

        # Autograd differentiates w.r.t. the first argument (index 0), which is `layers` [19].
        gradient_func = grad(cost_autograd, 0) 
        
        # Calculate gradients
        layers_grad = gradient_func(self.layers, inputs, targets)
        return layers_grad
        
