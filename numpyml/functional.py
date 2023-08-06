from numpyml.tensor import Tensor

def tanh(tensor):
    return tensor.tanh()

def cross_entropy(logits, target):
    return logits.cross_entropy(target)

def randn(*size, requires_grad=False):
    return Tensor.randn(*size, requires_grad=requires_grad)