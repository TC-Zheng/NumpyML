from numpyml.module import Module


class Sequential(Module):
    def __init__(self, *layers):
        self._layers = []
        for layer in layers:
            self._layers.append(layer)
    
    def forward(self, input):
        for layer in self._layers:
            input = layer(input)
        return input
    
    def parameters(self):
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params