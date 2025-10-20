import autograd.numpy as np
from autograd import grad, elementwise_grad

# --- Regression Loss and Gradient (Mean Squared Error) ---

def mse(yhat: np.ndarray, y: np.ndarray) -> float:
    """
    Mean Squared Error (MSE).
    C(Θ) = 1/N * sum((y_i - yhat_i)^2)
    """
    return np.mean((y - yhat) ** 2)

def dmse_dyhat(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of the MSE .
    dC/dyhat = (2/N) * (yhat - y)
    """
    n_samples = y.shape
    return (2.0 / n_samples) * (yhat - y)

# --- Classification Functions (Softmax, Cross-Entropy) ---

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Softmax activation function, commonly used in the output layer for 
    multiclass classification. Implemented with numerical stability.
    """
    # Subtract np.max(logits) for numerical stability 
    X_stable = logits - np.max(logits, axis=-1, keepdims=True)
    delta = 1e-15  # Small epsilon for robust division 
    
    exp_term = np.exp(X_stable)
    # Sum across the categories (axis=-1)
    probabilities = exp_term / (np.sum(exp_term, axis=-1, keepdims=True) + delta)
    return probabilities

def cross_entropy(yhat: np.ndarray, y_onehot: np.ndarray) -> float:
    """
    Categorical Cross-Entropy (CCE) loss function. This is the negative 
    log-likelihood for classification problems.
    
    NOTE: yhat should be the probabilities (output of softmax).
    """
    delta = 1e-15  # Numerical stability to avoid log(0) [8, 9, 13]
    yhat = np.clip(yhat, delta, 1.0 - delta)

    # Based on the CCE definition: -1/N * sum(y_i * log(yhat_i)) [14]
    return -(1.0 / y_onehot.shape) * np.sum(y_onehot * np.log(yhat))

def dCE_dlogits(yhat: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
    """
    Combined gradient of Cross-Entropy loss w.r.t. the input logits (z_L).
    This uses the simplification: dC/dz_L = (yhat - y_onehot) / N.
    """
    n_samples = yhat.shape
    # The error term (delta) for the output layer simplifies to (prediction - target) 
    # when using Softmax and Cross-Entropy [15, 16]. 
    # We divide by N (n_samples) to get the batch mean gradient as requested.
    return (yhat - y_onehot) / n_samples

# --- Regularization Penalties and Gradients ---

def l2_penalty(weights: np.ndarray, lmbd: float) -> float:
    """L2 regularization penalty: lambda * sum(W^2) [17]."""
    return lmbd * np.sum(weights ** 2)

def grad_l2(weights: np.ndarray, lmbd: float) -> np.ndarray:
    """Gradient of L2 penalty: adds 2 * lambda * W to the total gradient [18, 19]."""
    return 2.0 * lmbd * weights

def l1_penalty(weights: np.ndarray, lmbd: float) -> float:
    """L1 regularization penalty: lambda * sum(|W|) [10]."""
    return lmbd * np.sum(np.abs(weights))

def grad_l1(weights: np.ndarray, lmbd: float) -> np.ndarray:
    """Subgradient of L1 penalty: adds lambda * sign(W) [18]."""
    return lmbd * np.sign(weights)

# --- Loss Composition Helpers ---

def loss_regression(pred: np.ndarray, y: np.ndarray, l1_reg_sum: float = 0, l2_reg_sum: float = 0) -> float:
    """
    Calculates the total cost for a regression task (MSE + Regularization).
    The regularization sums (l1_reg_sum, l2_reg_sum) must be calculated 
    across all weights beforehand.
    """
    # C(Θ) = L(Θ) + λ||w||^2 [17]
    base_loss = mse(pred, y)
    return base_loss + l1_reg_sum + l2_reg_sum

def loss_classification(logits: np.ndarray, y_onehot: np.ndarray, l1_reg_sum: float = 0, l2_reg_sum: float = 0) -> float:
    """
    Calculates the total cost for a classification task (Cross-Entropy + Regularization).
    """
    probabilities = softmax(logits)
    # The cost function for classification is the negative log-likelihood (Cross-Entropy) [11, 12].
    base_loss = cross_entropy(probabilities, y_onehot)
    return base_loss + l1_reg_sum + l2_reg_sum