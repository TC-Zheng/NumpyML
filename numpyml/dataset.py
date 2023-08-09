
class Dataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        # Make sure data and labels have the same length
        if self.data.shape[0] != self.labels.shape[0]:
            raise ValueError("Data and labels must have the same length")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]