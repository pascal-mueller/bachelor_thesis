import h5py
import pycbc
import numpy as np
import torch

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, filename, stride=0.1, detector='H1', device='cuda'):
        self.device = device
        self.detector = detector
        self.sample_rate = 2048
        self.stride = stride
        self.step = self.sample_rate * self.stride

        # Init file
        self.file = h5py.File(filename, 'r')

        # Read datasets keys for given detector
        self.dataset_keys = self.file[detector].keys()
        self.start_times = []
        self.dataset_lengths = []
        
        self.datasets = []
        self.length = 0
        
        # Read actual datasets
        total = 0
        for i, key in enumerate(self.dataset_keys):
            # Read dataset i.e. segment
            dataset = self.file[detector][key]
            self.datasets.append(dataset)

            # Whiten dataset
            """
            print(f"Whitening dataset {i}")
            dataset = pycbc.types.TimeSeries(dataset, delta_t = 1.0 / self.sample_rate)
            dataset = dataset.whiten(0.5, 0.25, remove_corrupted = True,
                    low_frequency_cutoff = 18.0)
            dataset = dataset.numpy()

            # Store whitenset
            self.datasets.append(dataset)
            """

            # Set start times
            self.start_times.append(float(key))

            # Compute length in amount of samples of size self.stride
            num_seconds = len(dataset) / self.sample_rate
            total += num_seconds

            # We have a moving window of 1s, thus the - 1.
            length = int(((num_seconds - 1 ) / self.stride))
            self.length += length
            self.dataset_lengths.append(length)

        # Compute total duration: Add moving window for each dataset.
        self.duration = self.length * self.stride + len(self.datasets)

    def __len__(self):
        return self.length
    
    # TODO: If we have a stride of 0.1 then we get 2048*0.1 = 204.8 but we need
    # an integer. So currently we have some left over that at the end. Fix that.
    # => This should be fixed.

    # Always returns 1s of data
    def __getitem__(self, i):
        # To get the i-th sample, we first have to determine in which dataset
        # it is. Then we get it.

        # Loop over all datasets to find the one containing item i
        for (j, length) in enumerate(self.dataset_lengths):
            # Is the item i in the j-th dataset?
            if sum(self.dataset_lengths[0:j+1]) > i:
                # local i (i.e. the i-th element in the j-th dataset)
                i_loc = i - sum(self.dataset_lengths[0:j])

                #start = int(np.ceil(i_loc*self.step))
                start = i_loc * 204
                end = start + 2048

                item = self.datasets[j][start : end]
                
                # Compute time #TODO: Maybe move time a bit since the merger
                # usually isn't at the start but the time currently is the
                # start time of the current item 
                start_time = self.start_times[j] # dataset keys are start time

                #TODO:Compute dt correctly

                dt = (1.0 / self.sample_rate) * 204

                time = start_time + i*dt + 0.5 # 0.5 cause merger is on avg there.


                return time, item 










