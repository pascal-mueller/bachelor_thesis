import h5py
import numpy as np
import torch

class SamplesDataset(torch.utils.data.Dataset):
    def __init__(self, filename, device='cuda'):
        self.device = device

        # Init file
        self.file = h5py.File(filename, 'r')

        # Init datasets
        self.samples = self.file['samples']
        self.labels = self.file['samples_labels']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        # Several indices
        if isinstance(i, (list, np.ndarray)):
            # Get samples from the HF5 dataset
            samples = self.samples[i]

            # Initialize list for labels
            labels = []
            
            # Read labels from HF5 dataset. Those are either 1 or 0.
            # Replace 1 with [1.0, 0.0] and 0 with [0.0, 1.0]
            for label in self.labels[i]:
                if label == 1:
                    labels.append([1.0, 0.0]) # noise + signal
                else:
                    labels.append([0.0, 1.0]) # pure noise

            return np.array(samples), np.array(labels)

        # One index
        else:
            sample = self.samples[i]
            label = self.labels[i]
            
            if label == 1:
                label = [1.0, 0.0] # noise + signal
            else:
                label = [0.0, 1.0] # pure noise

            return np.array(sample), np.array(label)

