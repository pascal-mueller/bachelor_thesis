import numpy as np
import pycbc.distributions

class SignalSpace:
    def __init__(self, N, stride):
        # Set the seed for self.rng and self.sky_location_dist
        np.random.seed(2854)
        self.rng = np.random.default_rng()
        self.sky_location_dist = pycbc.distributions.sky_location.UniformSky()
        
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
            # Create distributions
            # TODO: If only 1 detector, can probably set all angles to 0.
            # Check paper.
            angles = self.rng.uniform(0.0, 2 * np.pi, 3)
            masses = self.rng.uniform(10.0, 50.0, 2)

            # Draw parameters
            declination, right_ascension = self.sky_location_dist.rvs()[0]
            
            waveform_kwargs = {}
            waveform_kwargs['approximant'] = 'SEOBNRv4_opt'
            waveform_kwargs['delta_t'] = 1.0 / self.sample_rate
            waveform_kwargs['f_lower'] = 18.0
            waveform_kwargs['mass1'] = max(masses)
            waveform_kwargs['mass2'] = min(masses)
            waveform_kwargs['coa_phase'] = angles[0]
            waveform_kwargs['inclination'] = angles[1]

            # Set members
            sample = {} 
            sample['index'] = self.idx + i
            sample['waveform_kwargs'] = waveform_kwargs
            sample['declination'] = declination
            sample['right_ascension'] = right_ascension
            sample['pol_angle'] = angles[2]

            params.append(sample)

        return params 

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
            params = self.generate_parameters(self.stride)
            self.idx += self.stride
        else:
            params = self.generate_parameters(rest) 
            self.idx = self.N 
        
        return params
