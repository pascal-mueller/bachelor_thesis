import h5py
import numpy as np
import torch

class SamplesDataset(torch.utils.data.Dataset):
    def __init__(self, filename, device='cuda'):
        self.device = device

        # Init file
        self.file = h5py.File(filename, 'r')

        # Init first
        self.samples = self.file['samples']
        self.labels = self.file['samples_labels']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        label = self.labels[i]
        
        if label == 1:
            label = [1.0, 0.0] # noise + signal
        else:
            label = [0.0, 1.0] # pure noise

        return np.array(sample), np.array(label)

