import numpy as np
import pycbc.distributions

class NoiseSpace:
    def __init__(self, N, stride):
        print("\n\n\n\n\nGenerating noise: ", N, "\n\n\n\n")
        # Amount of samples
        self.N = N
        self.sample_rate = 2048

        # The stride with which we step through the parameter space (used by iterator)
        self.stride = stride
        
        # Index to keep track where we are
        self.idx = 0
    
    def generate_parameters(self, N):
        params =[] 

        for i in range(N):
            # Set members
            sample = {} 
            sample['index'] = self.idx + i

            params.append(sample)

        return params 

    def __len__(self):
        # TODO:: This might be an issue because we won't have
        # len * stride amount of samples.
        # TODO: Maybe enforce self.N % self.stride == 0. See also __next__.
        return int(np.ceil( self.N / self.stride))

    def __iter__(self):
        return self
    
    def __next__(self):
        # Rest of "division".
        rest = self.N % self.stride
        
        # Raise exception if no samples left
        if self.idx == self.N:
            raise StopIteration
        
        # Make sure we won't increase idx out of bound. While python doesn't
        # care, I do.
        if self.idx + self.stride <= self.N:
            params = self.generate_parameters(self.stride)
            self.idx += self.stride
        else:
            params = self.generate_parameters(rest) 
            self.idx = self.N 
        
        return params
