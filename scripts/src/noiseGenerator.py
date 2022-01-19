import pycbc.psd, pycbc.noise
import numpy as np

class NoiseGenerator:
    def __init__(self, psd, file=None, duration=1.25):
        if file == None:
            self.file = None
        else:
            self.dataset = file['noise']
        
        self.psd = psd
        self.duration = duration # or 1.25?
        self.sample_rate = 2048

    def generate(self, params=None):
        # Generate several
        if params != None:
            data = np.zeros((len(params), int(self.duration * self.sample_rate)))
            
            for (i, param) in enumerate(params):
                noise = self.generate_noise()
                data[i] = noise
            
                idx = params[0]['index']
                k = len(params)
            
                self.dataset[idx : idx + k] = data
        # Generate one and return
        else:
            return self.generate_noise()

    def generate_noise(self):
        # Create PSD for detector
        """
        psd_length = int(0.5 * self.sample_rate * self.duration) + 1
        delta_f = 1.0 / self.duration

        psd_fn = pycbc.psd.analytical.aLIGOZeroDetHighPower
        psd = psd_fn(length=psd_length, delta_f=delta_f, low_freq_cutoff=18.0)
        """

        # Generate noise from PSD
        seed = int((time.time()* time.time() )% 200000 )
        noise_fn = pycbc.noise.gaussian.frequency_noise_from_psd
        noise = noise_fn(self.psd, seed).to_timeseries().numpy()
        
        #idx = param['index']
        #self.dataset[idx] = noise

        return noise
