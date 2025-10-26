import autograd.numpy as np

"""Based on exercises week 42 (exercise 8 structure). 
Implements a feed-forward neural network using batched inputs,
with weights and biases stored separately for clarity."""

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
    ):
        """
        Initializes the neural network architecture. Weights are created using 
        the batched approach (W shape: Input size, Output size) [3, 4].
        """
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        
        # Create layers (weights and biases)
        self.layers = self._create_layers_batch(network_input_size, layer_output_sizes)

    def _create_layers_batch(self, network_input_size, layer_output_sizes):
        """
        Creates layers based on the batched input approach (W shape: Input size, Output size) [3, 4].
        """
        layers = []
        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            # W shape: (Input size, Output size) - transposed compared to single input [3, 4]
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size
        return layers

    def predict(self, inputs):
        """
        Simple feed forward pass for batched inputs, returning the final activation [4, 8].
        """
        a = inputs
        # Iterate over stored layers and activation functions [4, 8]
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            # Z = a @ W + b. a is (N, I), W is (I, O), Z is (N, O) [9, 10]
            z = np.matmul(a, W) + b
            a = activation_func(z)
        return a

    def cost(self, inputs, targets):
        """
        Calculates the cost based on the network prediction and targets [7, 10].
        """
        predict_batch = self.predict(inputs)
        return self.cost_fun(predict_batch, targets) # Using Cost function definition [10]

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
        
        predict_batch = a
        return layer_inputs, zs, predict_batch # Returns saved data and final prediction [12]

    def compute_gradient(self, inputs, targets):
        """
        Computes the gradient of the cost function w.r.t. all weights and biases 
        using the backpropagation algorithm for batched inputs [10, 13].
        """
        batch_size = inputs.shape
        layer_inputs, zs, predict_batch = self._feed_forward_saver(inputs)
        
        layer_grads = [ (np.zeros_like(W), np.zeros_like(b)) for (W, b) in self.layers]
        
        # Loop backwards over the layers [13, 14]
        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]
            
            # 1. Compute delta (dC/dz)
            if i == len(self.layers) - 1:
                # Last layer: Special case (Softmax + Cross-Entropy simplifies to pred - target) [12, 15]. 
                # If using generic MSE, use cost_der * activation_der(z)
                
                # Check if it corresponds to the Softmax/Cross-Entropy simplification
                if self.activation_funcs[i].__name__ == "softmax" and self.cost_fun.__name__ == "CostCrossEntropy":
                     dC_dz = predict_batch - targets 
                else:
                    # Generic case (e.g., Regression or binary classification cost)
                    # Note: We assume cost_der takes (predict, target) and returns dC/da
                    dC_da = self.cost_der(predict_batch, targets) 
                    dC_dz = activation_der(z) * dC_da
            else:
                # Hidden layers: Backpropagate error from the next layer [13]
                W_next, _ = self.layers[i + 1]
                dC_da = dC_dz @ W_next.T
                dC_dz = activation_der(z) * dC_da
            
            # 2. Compute W and b gradients using delta [13]
            # dC/dW = (Input.T @ dC/dz) / batch_size [13]
            dC_dW = (layer_input.T @ dC_dz) / batch_size
            # dC/db = mean(dC/dz) [13]
            dC_db = np.mean(dC_dz, axis=0)

            layer_grads[i] = (dC_dW, dC_db)
            
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