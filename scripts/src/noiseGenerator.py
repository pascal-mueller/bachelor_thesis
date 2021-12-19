import pycbc.psd, pycbc.noise
import numpy as np

class NoiseGenerator:
    def __init__(self, file):
        self.dataset = file['noise']
        self.duration = 1.25 # or 1.25?
        self.sample_rate = 2048

    def generate(self, params):
        data = np.zeros((len(params), 2560))

        for (i, param) in enumerate(params):
            # Create PSD for detector
            psd_length = int(0.5 * self.sample_rate * self.duration) + 1
            delta_f = 1.0 / self.duration

            psd_fn = pycbc.psd.analytical.aLIGOZeroDetHighPower
            psd = psd_fn(length=psd_length, delta_f=delta_f, low_freq_cutoff=18.0)

            # Generate noise from PSD
            noise_fn = pycbc.noise.gaussian.frequency_noise_from_psd
            noise = noise_fn(psd).to_timeseries().numpy()
            
            #idx = param['index']
            #self.dataset[idx] = noise
            data[i] = noise
        
        idx = params[0]['index']
        k = len(params)

        self.dataset[idx : idx + k] = data
