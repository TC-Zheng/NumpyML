from numpyml.models import Sequential
from numpyml.layers import Linear, Tanh
from numpyml import Tensor
import numpyml.functional as F
from mnist_data_loader import MnistDataloader
import numpy as np
from matplotlib import pyplot as plt

# Load MINST dataset
mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

model = Sequential(Linear(784, 100), Tanh(), Linear(100, 10))

# Create a batch of 20 images
batch_size = 20
input = x_train[0:batch_size]
# flatten the images
for i in range(batch_size):
    input[i] = np.concatenate([element for element in input[i]])
x = Tensor(input)
y = Tensor(y_train[0:batch_size])
result = model(x)
loss = F.cross_entropy(result, y)
print(loss)
loss.backward() 