from numpyml.tensor import Tensor
import numpy as np

def tanh(tensor):
    return tensor.tanh()

def cross_entropy(logits, target):
    return logits.cross_entropy(target)

def randn(*size, requires_grad=False):
    return Tensor.randn(*size, requires_grad=requires_grad)

def one_hot(target, num_classes):
    # One-hot encoding only works for 1D tensors
    if len(target.shape) != 1:
        raise ValueError("One-hot encoding only works for 1D tensors.")
    # Initialize the one-hot matrix
    one_hot = np.zeros((target.shape[0], num_classes))
    # Using advanced indexing to fill in the one-hot matrix
    one_hot[np.arange(target.shape[0]), target] = 1
    return Tensor(one_hot)

def zeros(*size, requires_grad=False):
    return Tensor(np.zeros(*size), requires_grad=requires_grad)