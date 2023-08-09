from numpyml import Tensor
import numpy as np

class Dataset:
    def __init__(self, data, labels):
        # Make sure data and labels have the same length
        if data.shape[0] != labels.shape[0]:
            raise ValueError("Data and labels must have the same length")
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]