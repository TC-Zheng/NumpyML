import pytest
from numpyml import Tensor
import numpyml.functional as F
import numpy as np
import pickle
# pytest --cov=/home/tc_zheng/projects/NumpyML/numpyml for coverage
# pytest --cov=/home/tc_zheng/projects/NumpyML/numpyml --cov-report html for html report

def test_tensor_initialization():
    
    t = F.randn(3, 3)
    assert t.shape == (3, 3)
    assert t.dtype == np.float64    
    
def test_tensor_arithmatics():
    
    a = Tensor([1, 2, 3, 4, 5])
    b = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    assert (a == b) == Tensor([True, True, True, True, True])
    
    assert (a + b) == Tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    
    assert (a - b) == Tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    
    assert (a * b) == Tensor([1.0, 4.0, 9.0, 16.0, 25.0])
    
    assert np.allclose(F.tanh(a).data, np.array([0.76159416, 0.96402758, 0.99505475, 0.9993293 , 0.9999092 ]))
    
def test_cross_entropy():
    # Tests tensor multiplication, addition, cross entropy loss, and backpropagation
    w1 = Tensor([[1.0,2.0,3.0], [4.0,5.0,6.0]], requires_grad=True) # w1.shape = (2,3)
    w2 = Tensor([[9.0,8.0,7.0], [6.0,5.0,4.0], [3.0,2.0,1.0]], requires_grad=True) # w2.shape = (3,3)
    b = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    logits = w1 @ w2 + b # w3.shape = (batch_size, 3)
    labels = Tensor([1, 0]) # labels.shape = (batch_size)
    # Compute the cross entropy loss
    loss = F.cross_entropy(logits, labels)
    # Compute the gradients of loss with respect to w1 and w2
    loss.backward()
    # Load the expected results from the pickle file
    with open('tests/results.pkl', 'rb') as f:
        expected_results = pickle.load(f)
    results = [w1.grad, w2.grad, b.grad, loss.data]
    # Compare the results
    for result, expected_result in zip(results, expected_results):
        assert np.allclose(result, expected_result)
        
    def test_backward_error():
        # Backward should only be called on a scalar tensor.
        x = Tensor([1, 2, 3, 4, 5])
        with pytest.raises(ValueError) as excinfo:
            x.backward()
        assert excinfo.match("Backward can should be called on a scalar tensor.")