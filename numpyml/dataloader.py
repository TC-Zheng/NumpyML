import numpy as np
from numpyml import Tensor

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Create a list of indices for all the data points
        indices = np.arange(len(self.dataset))
        # Shuffle the indices if necessary
        if self.shuffle:
            np.random.shuffle(indices)
        # Yield batches of the specified size
        for start_idx in range(0, len(self.dataset), self.batch_size):
            # The end index is the start index plus the batch size, or the end of the dataset,
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            # Get the indices for the batch
            batch_indices = indices[start_idx:end_idx]
            # Get the data and labels for the batch
            batch_data, batch_labels = zip(*[self.dataset[i] for i in batch_indices])
            yield np.array(batch_data), np.array(batch_labels)