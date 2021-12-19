import numpy as np

class SampleSpace:
    def __init__(self, N_noise, N_signal, N_samples, stride):
        self.rng = np.random.default_rng(33)
        
        self.N_noise = N_noise
        self.N_signal = N_signal
        self.N_samples = N_samples
        self.N = N_samples + N_noise

        self.idx_noise = self.rng.integers(0, N_noise, N_samples) 
        self.idx_signal = self.rng.integers(0, N_signal, N_samples)

        # The stride with which we step through the parameter space (used by iterator)
        self.stride = stride
        
        # Index to keep track where we are
        self.idx = 0

        self.samples = self.generate_samples()
    
    def generate_samples(self):
        idx_samples = np.ones(self.N_samples + self.N_noise, dtype=int)
        
        random_indices = self.rng.choice(self.N_samples + self.N_noise,
                size=self.N_noise, replace=False)
        
        idx_samples[random_indices] = 0 # Noise

        samples = []
        idx_noise = 0
        j = 0
        for (i, item) in enumerate(idx_samples):
            sample = {} 
            sample['index'] = i

            # signal + noise: Take signal and noise smaple using idx_...
            if item == 1: 
                sample['idx_noise'] = self.idx_noise[j]
                sample['idx_signal'] = self.idx_signal[j]
                j += 1
            # Pure noise: we shuffle in the pure noise
            else:
                sample['idx_noise'] = idx_noise 
                sample['idx_signal'] = None

                idx_noise += 1

            samples.append(sample)

        return samples 

    def __len__(self):
        # TODO:: This might be an issue because we won't have
        # len * stride amount of samples.
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
            params = self.samples[self.idx : self.idx + self.stride]
            self.idx += self.stride
        else:
            params = self.samples[self.idx : ]
            self.idx = self.N 
        
        return params
