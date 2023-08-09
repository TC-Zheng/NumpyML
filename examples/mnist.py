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
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

# Load the training labels.
y_train = y_train
y_test = y_test

# Create a dataset and a dataloader
train_dataset = Dataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)

# Create a model
model = Sequential(Linear(784, 200), Tanh(), Linear(200, 200), Tanh(), Linear(200, 10))

# Create an optimizer
optimizer = SGD(model.parameters(), lr=0.005)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    for batch_x, batch_y in train_dataloader:
        # Forward pass
        logits = model(Tensor(batch_x))
        # Compute the cross entropy loss
        loss = F.cross_entropy(logits, Tensor(batch_y))
        # Backward pass
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Zero out the gradients
        optimizer.zero_grad()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.data))
    

# Test the accuracy of the model against the test dataset
test_logits = model(Tensor(x_test))
predictions = np.argmax(test_logits.data, axis=1)
accuracy = np.mean(predictions == y_test)
print('Test set accuracy: {:.2f}%'.format(accuracy * 100))
        