import numpy as np
from tensor import Tensor
from models import MLP
# ts = []

# # ts.append(Tensor([1, 2, 3], requires_grad=True))
# # ts.append(Tensor([4, 5, 6], requires_grad=True))
# # ts.append(Tensor([7, 8, 9], requires_grad=True))
# # ts.append(Tensor([10, 11, 12], requires_grad=True))
# # result = ts[0] + ts[1] * ts[2]
# # result = result * ts[3]
# # result.backward()
# ts.append(Tensor([[1, 2, 3], [4,5,6], [7,8,9]], requires_grad=True))
# result = ts[0] + ts[0]
# result.backward()

# for tensor in ts:
#     print('data:', tensor.data)
#     print('prev:', tensor._prev)
#     print('operation:', tensor._operation)
#     print('grad:', tensor.grad)
#     print('-' * 20)

# logits = Tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]], requires_grad=True)
# target_labels = Tensor([2, 0])
# print(logits.data.shape)
# print(target_labels.data.shape)

# loss = logits.cross_entropy(target_labels)
# loss.backward()
# print(loss.data)
# print(logits.grad)  

# w1 = Tensor([[1.0,2.0], [3.0, 4.0]])
# w2 = Tensor([[5.0,6.0], [7.0, 8.0]])
# logits = w1 @ w2
# print(logits.data.shape)
# target_labels = Tensor([1, 0])
# print(target_labels.data.shape)
# loss = logits.cross_entropy(target_labels)
# print(loss)
# loss.backward()
# print(w1.grad)
# print(w2.grad)

model = MLP([3, 4, 2])
out = model(Tensor([[1, 2, 3], [4, 5, 6]]))
print(out)