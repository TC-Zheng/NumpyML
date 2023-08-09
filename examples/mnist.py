from numpyml.models import Sequential
from numpyml.layers import Linear, Tanh
from numpyml import Tensor
import numpyml.functional as F
from mnist_data_loader import MnistDataloader
import numpy as np
from matplotlib import pyplot as plt
from numpyml.dataset import Dataset
from numpyml.dataloader import DataLoader
from numpyml.optimizers import SGD

# Load MINST dataset
mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Reshape the input image tensors to vectors, and normalize the values to be between 0 and 1
x_train = Tensor(x_train.reshape(60000, 784) / 255)
x_test = Tensor(x_test.reshape(10000, 784) / 255)

# Load the training labels. Convert the labels to one-hot encoded vectors
y_train = Tensor(F.one_hot(y_train, 10))
y_test = Tensor(F.one_hot(y_test, 10))

# Create a dataset and a dataloader
train_dataset = Dataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# Create a model
model = Sequential(Linear(784, 100), Tanh(), Linear(100, 10))

# Create an optimizer
optimizer = SGD(model.parameters(), lr=0.05)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for batch_x, batch_y in train_dataloader:
        # Forward pass
        logits = model(batch_x)
        # Compute the cross entropy loss
        loss = F.cross_entropy(logits, batch_y)
        print(loss)
        # Backward pass
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Zero out the gradients
        optimizer.zero_grad()
        