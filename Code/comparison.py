"""Helper for running the comparisons etc"""

from math import ceil
from pathlib import Path
import sys


here = Path.cwd()
candidates = [here] + list(here.parents)
for p in candidates:
    if (p / "Code").is_dir():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Couldn't find a 'Code' folder in this project.")

import autograd.numpy as np
from Code.ffnn2 import NeuralNetwork as FFNN 
from Code.activations import sigmoid, identity, derivate 
from Code.cost import CostOLS, dCostOLS, CostCrossEntropy, dCostCrossEntropy, softmax



#Handles the cost functions as well
def run_comparison(optimizer, X_train, t_train, network_input_size, layer_output_sizes, **kwargs):
    # Determine activation functions based on layers count
    n_hidden_layers = len(layer_output_sizes) - 1
    
    # Sigmoid for hidden layers, Identity for output (Regression)
    activation_funcs = [sigmoid] * n_hidden_layers + [identity]
    activation_ders = [derivate(f) for f in activation_funcs] # Derivatives of the activation functions
    
    # Initialize NN 

    nn = FFNN(
    network_input_size=network_input_size, 
    layer_output_sizes=tuple(layer_output_sizes),
    activation_funcs=activation_funcs,  # List/tuple of all activation functions (hidden + output) 
    activation_ders=activation_ders,    # List/tuple of all corresponding derivatives 
    cost_fun=CostOLS,                  # Cost function 
    cost_der=dCostOLS, 
    seed=42
    )

    nn.reset_weights()

    # Extract common nn.fit arguments from kwargs (epochs, batches, lam) 
    epochs = kwargs.pop('epochs', 1000)
    batches = kwargs.pop('batches', 32)
    lam = kwargs.pop('lam', 0.0)
    l1  = kwargs.pop('l1', None)          
    l2  = kwargs.pop('l2', None) 

    # Instantiate the scheduler using remaining kwargs (eta, rho, rho2, etc.)
    opt_args = {k: v for k, v in kwargs.items() if k in ("eta", "rho", "rho2", "eps", "beta1", "beta2")}
    scheduler_instance = optimizer(**opt_args)

    n_train  = X_train.shape[0]
    batches  = ceil(n_train / 64)

    # Train the network
    scores = nn.fit(
        X_train, 
        t_train, 
        scheduler=scheduler_instance, 
        epochs=epochs, 
        batches=batches,
        lam=lam,
        l1=l1,
        l2=l2
    )
    return scores, nn



"""Classification comparison helper functions."""

def run_classification_comparison(optimizer, X_train, Y_train, X_test, y_test,
                                   network_input_size, layer_output_sizes, hidden_activation_func, **kwargs):
    n_hidden = len(layer_output_sizes) - 1
    activation_funcs = [hidden_activation_func] * n_hidden + [softmax]  # Hidden + output softmax
    activation_ders = [derivate(f) for f in activation_funcs]  # Derivatives of activations

    nn = FFNN(network_input_size=network_input_size,
              layer_output_sizes=tuple(layer_output_sizes),
              activation_funcs=activation_funcs,
              activation_ders=activation_ders,
              cost_fun=CostCrossEntropy,
              cost_der=dCostCrossEntropy,
              seed=42)
    nn.reset_weights()

    epochs = kwargs.pop('epochs', 10)
    batches = kwargs.pop('batches', 64)
    lam = kwargs.pop('lam', 0.0)
    l1 = kwargs.pop('l1', 0.0)
    l2 = kwargs.pop('l2', 0.0)

    opt_args = {k: v for k, v in kwargs.items() if k in ("eta", "rho", "rho2", "eps", "beta1", "beta2")}
    scheduler = optimizer(**opt_args)

    # Training the network
    history = nn.fit(X_train, Y_train, scheduler=scheduler, epochs=epochs, batches=batches,
                     lam=lam, l1=l1, l2=l2)

    # Testing
    probs_test = nn.predict(X_test)
    y_hat = np.argmax(probs_test, axis=1)
    acc = (y_hat == y_test).mean()  # Accuracy
    ce_test = CostCrossEntropy(probs_test, np.eye(10)[y_test])  # Cross entropy cost on test data

    return {"train_errors": history["train_errors"], "acc": acc, "ce": ce_test}, nn