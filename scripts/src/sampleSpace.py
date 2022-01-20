import numpy as np

class SampleSpace:
    def __init__(self, N_noise, N_signal, stride):
        self.rng = np.random.default_rng(33)
        
        self.N_noise = N_noise
        self.N_signal = N_signal
        self.N = N_noise + N_signal

        self.idx_noise = np.arange(self.N_noise + self.N_signal)
        self.idx_signals = np.arange(self.N_signal)
        np.random.shuffle(self.idx_noise)
        np.random.shuffle(self.idx_signals)

        #self.idx_noise = self.rng.integers(0, N_noise, N_samples) 
        #self.idx_signal = self.rng.integers(0, N_signal, N_samples)

        # The stride with which we step through the parameter space (used by iterator)
        self.stride = stride
        
        # Index to keep track where we are
        self.idx = 0

        self.samples = self.generate_samples_params()
    
    # NOTE: We have N_noise pure noise samples and N_signal pure signals.
    # For samples, we gonna inject all N_signals into a noise, so we actually
    # have N_noise + N_signal noises in total.
    #
    # N_noise + N_signal amount of "noise+signal" samples.
    # N_noise amount of "noise" samples.
    #
    # So we get N_noise + N_signal amount of samples.
    def generate_samples_params(self):
        # Generate an array full of ones
        idx_samples = np.ones(self.N_signal + self.N_noise, dtype=int)
        
        # Choose N_noise amount of indices randomly
        random_indices = self.rng.choice(self.N_noise + self.N_signal,
                size=self.N_noise, replace=False)
        
        # Set the randomly chosen indices to 0. We end up with an array of
        # length N_noise + N_signal where:
        # 0: pure noise
        # 1: noise+signal
        idx_samples[random_indices] = 0 # Noise
        
        samples = []
        j = 0
        for (i, item) in enumerate(idx_samples):
            sample = {} 
            sample['index'] = i

            # signal + noise: Take signal and noise smaple using idx_...
            if item == 1: 
                sample['idx_noise'] = self.idx_noise[i]
                sample['idx_signal'] = self.idx_signals[j]
                j += 1
            # Pure noise: we shuffle in the pure noise
            else:
                sample['idx_noise'] = self.idx_noise[i]
                sample['idx_signal'] = None

            samples.append(sample)

        return samples 

    def __len__(self):
        # TODO:: This might be an issue because we won't have
        # len * stride amount of samples.
        # TODO: Enforce self.N % self.stride
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
