from tensor import Tensor

def tanh(tensor):
    return tensor.tanh()

def cross_entropy_loss(logits, target):
    return logits.cross_entropy_loss(target)

def randn(*size, requires_grad=False):
    return Tensor.randn(*size, requires_grad=requires_grad)