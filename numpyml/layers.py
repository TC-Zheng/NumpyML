from numpyml.module import Module
import numpyml.functional as F
import math

class Linear(Module):
    def __init__(self, fan_in, fan_out):
        self.weights = F.randn(fan_in, fan_out, requires_grad=True) # Random initialization
        self.bias = F.randn(fan_out, requires_grad=True) # Random initialization
        # self.bias = F.zeros(fan_out, requires_grad=True) # Zero initialization
        
        # Xavier initialization
        self.weights.data *= math.sqrt(2.0 / (fan_in + fan_out))
        self.bias.data *= math.sqrt(2.0 / (fan_in + fan_out))

    def forward(self, input):
        return input @ self.weights + self.bias
    
    def parameters(self):
        return [self.weights, self.bias]


class Tanh(Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        return x.tanh()
    
    def parameters(self):
        return []