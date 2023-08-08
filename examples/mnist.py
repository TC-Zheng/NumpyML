from numpyml.models import Sequential
import numpyml.functional as F
from mnist_data_loader import MnistDataloader
import os
import random
from matplotlib import pyplot as plt

# Load MINST dataset
mnist_dataloader = MnistDataloader()
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
