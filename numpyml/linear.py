from numpyml.module import Module
import numpyml.functional as F

class Linear(Module):
    def __init__(self, fan_in, fan_out, activation='tanh'):
        self.weights = F.randn(fan_in, fan_out, requires_grad=True) # Random initialization
        self.bias = F.randn(fan_out, requires_grad=True) # Random initialization
        self._activation = activation
        
    def forward(self, input):
        if self._activation == 'tanh':
            return F.tanh(input @ self.weights + self.bias)
