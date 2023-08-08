from abc import abstractmethod

class Module:
    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def __call__(self, input):
        return self.forward(input)