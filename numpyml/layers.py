from numpyml.module import Module
import numpyml.functional as F

class Linear(Module):
    def __init__(self, fan_in, fan_out):
        self.weights = F.randn(fan_in, fan_out, requires_grad=True) # Random initialization
        self.bias = F.randn(fan_out, requires_grad=True) # Random initialization
        
    def forward(self, input):
        return input @ self.weights + self.bias


class Tanh(Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        return x.tanh()