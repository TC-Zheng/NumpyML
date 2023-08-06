from numpyml.module import Module
from numpyml.linear import Linear

class MLP(Module):
    def __init__(self, layer_sizes, activation='tanh'):
        self._layers = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            self._layers.append(Linear(fan_in, fan_out, activation))
        
    def forward(self, input):
        for layer in self._layers:
            input = layer(input)
        return input
        