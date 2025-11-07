from autograd import numpy as np

"""Cost functions"""

def CostOLS(pred, target):
    """Mean squared error over the batch."""
    return np.mean((pred - target) ** 2)

def dCostOLS(pred, target):
    """Derivative of MSE wrt predictions (averaged over batch)."""
    N = pred.shape[0]
    return (2.0 / N) * (pred - target)


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)  # stability
    expz = np.exp(logits)
    return expz / np.sum(expz, axis=1, keepdims=True)

def CostCrossEntropy(pred, targets):   # pred: probs (N,C); targets: one-hot (N,C)
    eps = 1e-12
    return -np.mean(np.sum(targets * np.log(pred + eps), axis=1))

def dCostCrossEntropy(pred, targets):  # softmax + CE shortcut
    N = pred.shape[0]
    return (pred - targets) / N