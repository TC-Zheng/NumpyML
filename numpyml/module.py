from abc import abstractmethod

class Module:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, input):
        pass
    
    def __call__(self, input):
        return self.forward(input)