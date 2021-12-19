import h5py
import numpy as np
import torch

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, filename, stride=0.1, detector='H1', device='cuda'):
        self.device = device
        self.detector = detector
        self.sample_rate = 2048
        self.stride = stride

        # Init file
        self.file = h5py.File(filename, 'r')

        # Read datasets keys for given detector
        self.dataset_keys = self.file[detector].keys()
        self.dataset_lengths = []
        
        self.datasets = []
        self.length = 0
        
        # Read actual datasets
        for key in self.dataset_keys:
            dataset = self.file[detector][key]
            self.datasets.append(dataset)

            # Compute length in amount of samples of size self.stride
            num_seconds = len(dataset) / self.sample_rate
            # We have a moving window of 1s, thus the - 1.
            length = int(((num_seconds - 1 ) / self.stride))
            self.length += length
            self.dataset_lengths.append(length)

        # Compute total duration: Add moving window for each dataset.
        self.duration = self.length * self.stride + len(self.datasets)


    def __len__(self):
        return self.length
    
    # TODO: If we have a stride of 0.1 then we get 2048/10 = 204.8 but we need
    # an integer. So currently we have some left over that at the end. Fix that.
    # Always returns 1s of data
    def __getitem__(self, i):
        k = 0

        for (j, length) in enumerate(self.dataset_lengths):
            if k + length > i:
                i = i * (self.sample_rate * self.stride)
                k = k * (self.sample_rate * self.stride)

                start = int(i - k)
                end = int(start + 2048)

                item = self.datasets[j][start : end]
                item = torch.tensor(item, device=self.device)

                return item
            else:
                k += length
