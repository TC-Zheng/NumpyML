import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:            
            # If the param.data is batched, we need to average the gradients
            if param.grad.ndim != param.data.ndim:
                param.grad = np.mean(param.grad, axis=(0, 1))
                
            # Update the parameters
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)  