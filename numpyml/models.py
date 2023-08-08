from numpyml.module import Module
from numpyml.layers import Linear


class Sequential(Module):
    def __init__(self, *modules):
        self._modules = []
        for module in modules:
            self._modules.append(module)
    
    def forward(self, input):
        for module in self._modules:
            input = module(input)
        return input