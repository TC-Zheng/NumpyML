import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _operation=''):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        # _prev consists of previous tensors that are used to compute the current tensor
        self._prev = _children
        # _operation is the operation that was used to compute the current tensor
        self._operation = _operation
        # default gradient is 0
        self.grad = None if not requires_grad else np.zeros_like(data)
        
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Tensor({self.data})"
    
    
    def backward(self):
        # Backward shouldn't be called on a tensor that doesn't require gradients
        if self.grad is None:
            raise ValueError("Backward shouldn't be called on a tensor that doesn't require gradients.")
        # Ensure that the current tensor is a scalar
        if np.size(self.data) != 1:
            raise ValueError("Backward should only be called on a scalar tensor.")
        # Topological sort
        topo = []
        visited = set()
        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for t in tensor._prev:
                    build_topo(t)
                topo.append(tensor)
        build_topo(self)
        # The grad was default to 0, so we need to set it to 1
        self.grad = np.ones_like(self.data)
        for tensor in reversed(topo):
            tensor._backward()
            
    def __eq__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data == other.data)
        return False

    def __hash__(self):
        return hash(self.data.tobytes())
    
    def __add__(self, other):
        """Adds two tensors entrywise. The resulting tensor requires gradient computation 
    if either of the input tensors does. During the backward pass, gradients 
    are propagated to input tensors that require gradients and not propagated 
    to those that do not.
        """
        requires_grad = self.grad is not None or other.grad is not None
        out = Tensor(self.data + other.data, requires_grad, (self, other), '+')
        def _backward():
            # The gradient simply passes through for addition
            if self.grad is not None:
                self.grad += out.grad
            if other.grad is not None:
                other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """Multiplies two tensors entrywise. The resulting tensor requires gradient computation 
    if either of the input tensors does. During the backward pass, gradients 
    are propagated to input tensors that require gradients and not propagated 
    to those that do not.
        """
        requires_grad = self.grad is not None or other.grad is not None
        out = Tensor(self.data * other.data, requires_grad, (self, other), '*')
        def _backward():
            # Derivative for multiplication
            if self.grad is not None:
                self.grad += other.data * out.grad
            if other.grad is not None:
                other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        """Subtracts two tensors entrywise. The resulting tensor requires gradient computation
    if either of the input tensors does. During the backward pass, gradients
    are propagated to input tensors that require gradients and not propagated
    to those that do not.
        """
        # Subtraction is just addition with the second tensor negated
        requires_grad = self.grad is not None or other.grad is not None
        out = Tensor(self.data - other.data, requires_grad, (self, other), '-')
        def _backward():
            if self.grad is not None:
                self.grad += out.grad
            if other.grad is not None:
                other.grad -= out.grad
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        """Multiplies two tensors using matrix multiplication. The resulting tensor requires
    gradient computation if either of the input tensors does. During the backward pass,
    gradients are propagated to input tensors that require gradients and not propagated
    to those that do not."""
        requires_grad = self.grad is not None or other.grad is not None
        out = Tensor(self.data @ other.data, requires_grad, (self, other), '@')
        def _backward():
            # Gradients for matrix multiplication
            if self.grad is not None:
                self.grad += out.grad @ other.data.T
            if other.grad is not None:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def tanh(self):
        """Hyperbolic tangent function."""
        requires_grad = self.grad is not None
        out = Tensor(np.tanh(self.data), requires_grad, (self,), 'tanh')
        def _backward():
            # Derivative for tanh
            self.grad = (1 - np.tanh(self.data) ** 2) * out.grad
        out._backward = _backward
        return out
    
    def cross_entropy(self, target):
        """
    Compute the cross-entropy loss between logits and target classes.

    This function takes in logits (unnormalized scores for each class) and target class indices
    and returns the cross-entropy loss, averaged over the batch. The logits are assumed to be
    a 2D array of shape [batch size, number of classes], and the targets are assumed to be a
    1D array of shape [batch size].

    Parameters:
        target (Tensor): A tensor containing the target class indices for the batch.
                         It must be a 1D array of shape [batch size], with each element
                         being an integer representing the target class index.

    Returns:
        Tensor: A tensor containing the average cross-entropy loss over the batch. The result
                will require gradients if either input requires gradients.

    Raises:
        ValueError: If the input logits are not a 2D array of shape [batch size, number of classes],
                    or if the target is not a 1D array of shape [batch size].

    Example:
        logits = Tensor([[0.2, 0.4, 0.4], [0.1, 0.6, 0.3]], requires_grad=True)
        targets = Tensor([1, 2], requires_grad=False)
        loss = logits.cross_entropy(targets)
    """
        # Ensure that self.data represents logits and has the correct shape
        if self.data.ndim != 2:
            raise ValueError("Input must be of the form [batch size, number of classes]")
        if target.data.ndim != 1 or target.data.shape[0] != self.data.shape[0]:
            raise ValueError("Target must be of the form [batch size]")
        if self.data.shape[1] < 2:
            raise ValueError("Must have at least 2 classes")

        # Compute the softmax of the logits
        # Subtracts the maximum value in each row to improve numerical stability
        exps = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
        softmax = exps / np.sum(exps, axis=1, keepdims=True)

        # Select the values corresponding to the target indices
        batch_size = self.data.shape[0]
        target_values = softmax[np.arange(batch_size), target.data.astype(int)]

        # Compute the negative log likelihood
        loss = -np.log(target_values)

        # Compute the average loss over the batch
        average_loss = np.mean(loss)

        # Create a Tensor for the loss, with requires_grad=True if either input requires gradients
        requires_grad = self.grad is not None or target.grad is not None
        out = Tensor(average_loss, requires_grad, (self, target), 'cross_entropy')

        def _backward():
            # Compute the gradient with respect to the logits
            softmax[np.arange(batch_size), target.data.astype(int)] -= 1
            grad_logit = softmax / batch_size
            self.grad += grad_logit * out.grad
            # Gradient with respect to target is undefined, so it remains None

        out._backward = _backward
        return out
    
    def randn(*size, requires_grad=False):
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            size = size[0]
        return Tensor(np.random.randn(*size), requires_grad=requires_grad)
