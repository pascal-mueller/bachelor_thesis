import h5py
import pycbc.detector, pycbc.waveform, pycbc.types
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.signalSpace import SignalSpace
from src.signalGenerator import SignalGenerator
from src.noiseGenerator import NoiseGenerator 

class SingleTestDataset(torch.utils.data.Dataset):
    def __init__(self, stride=0.1, duration=10, device='cuda'):
        self.sample_rate = 2048
        self.duration = duration
        self.stride = stride
        self.rng = np.random.default_rng()

        self.length = self.duration * self.sample_rate
        self.noise = self.get_noise()
        self.signals = self.get_signals(N=6)
        self.strain = self.get_strain()

    def get_noise(self):
        noise_gen = NoiseGenerator(duration = self.duration)

        return noise_gen.generate() 

    
    def get_signals(self, N):
        signal_space = SignalSpace(N=N, stride=1)
        signal_gen = SignalGenerator()

        signals = np.zeros(self.length)
        k = int(self.length / (N+1) )

        print("length=", self.length, " k=", k)

        for i, signal_params in enumerate(signal_space):
            signal = signal_gen.generate(signal_params)[0]
            
            start = (i+1) * int( k - 0.5*len(signal) )
            end = start + len(signal)
            signals[start : end] = signal

            print("Inject at", start)

        return signals
    
    def get_strain(self):
        n_noise = len(self.noise)
        n_signal = len(self.signals)

        k_inj = int(0.5 * (n_noise - n_signal))

        strain = self.noise.copy()
        
        # Scale SNR and also make injection
        strain = self.SNR_scale()
        strain = self.whiten(strain)
        
        return strain

    def SNR_scale(self):
        # Create PSD for detector
        psd_length = int(0.5 * self.sample_rate * self.duration) + 1
        delta_f = 1.0 / self.duration
        psd_fn = pycbc.psd.analytical.aLIGOZeroDetHighPower
        psd = psd_fn(length=psd_length, delta_f=delta_f, low_freq_cutoff=18.0)
    
        # df = 1 / duration
        # dt = 1 / sample_rate
        signal = pycbc.types.TimeSeries(self.signals, 
                             delta_t = 1.0 / self.sample_rate, dtype=np.float64)
        foo = pycbc.filter.matchedfilter.sigmasq(signal, psd=psd,
                low_frequency_cutoff = 18.0)
        network_snr = np.sqrt(foo)
        target_snr = self.rng.uniform(5.0, 15.0)
        
        # TODO: Understand snr scaling here
        self.signals = signal.numpy() * (target_snr/network_snr)
        sample = self.noise + self.signals
        """
        fix, axs = plt.subplots(4, 1)
        axs[0].plot(range(len(self.noise)), self.noise)
        axs[1].plot(range(len(signal)), signal)
        axs[2].plot(range(len(self.noise)), self.noise)
        axs[2].plot(range(len(signal)), signal)
        axs[3].plot(range(len(sample)), sample)
        plt.show()
        """
        return sample
    
    def whiten(self, strain):
        # Whiten
        strain = pycbc.types.TimeSeries(strain, delta_t = 1.0 / self.sample_rate)
        # TODO: How coose params for whiten?
        # TODO: After whitening we only have 1s left. Input was 1.5s.
        # How do we get exaclty 1s?
        # ASSUMING 1.25 s
        strain = strain.whiten(0.5, 0.25, remove_corrupted = False,
                low_frequency_cutoff = 18.0)
    
        return strain.numpy()

    def plot(self, probabilities=[]):
        # "time"
        t = range(len(self.strain))
        
        if len(probabilities) == 0:
            fig, axs = plt.subplots(2, 1)
        else:
            fig, axs = plt.subplots(3, 1)

        
        # Plot noise with signal overlayed
        axs[0].plot(t, self.noise)
        axs[0].plot(t, self.signals)
        axs[0].set_title("Pure noise with signal overlayed")

        # Plot strain (noise + signal)
        axs[1].plot(t, self.strain)
        axs[1].set_title("SNR scaled & whitened strain (Noise + signal)")

        if len(probabilities) > 0:
            axs[2].plot(t, probabilities)
            axs[2].set_title("Probability signal")
        
        print("Plotting...")
        plt.show()

    def __len__(self):
        length = ( self.duration - 1 ) / self.stride + 1

        return int(length)

    def __getitem__(self, i):
        start = i * ( self.sample_rate * self.stride )
        start = int(start)

        end = start + 2048

        return start, end, np.array(self.strain[start : end])
"""


class SingleTestDataset(torch.utils.data.Dataset):
    def __init__(self, filename, stride=0.1, device='cuda'):
        self.device = device
        self.sample_rate = 2048
        self.stride = stride

        # Init file
        self.file = h5py.File(filename, 'r')
        
        self.noise_ds = self.file['noise']
        self.signals_ds = self.file['signals']
        self.samples_ds = self.file['samples']
        self.labels_ds = self.file['samples_labels']

        self.length = 0
        
        # Compute length in amount of samples of size self.stride
        num_seconds = ( len(self.samples_ds) - len(self.noise_ds) ) / self.stride
        print(num_seconds)
        # We have a moving window of 1s, thus the - 1.
        self.length = int(((num_seconds - 1 ) / self.stride))
        print(self.length)

        # Compute total duration: Add moving window for each dataset.
        self.duration = self.length * self.stride + 1 

    def __len__(self):
        return self.length

    def get_signal_and_noise(self):
        return self.samples_ds[0], self.signals_ds[0][0:2048], self.noise_ds[0][0:2048]
    
    # TODO: If we have a stride of 0.1 then we get 2048*0.1 = 204.8 but we need
    # an integer. So currently we have some left over that at the end. Fix that.
    # Always returns 1s of data
    def __getitem__(self, i):
        p = i
        k = 0

        # To get the i-th sample, we first have to determine in which dataset
        # it is. Then we get it.
        i = i * (self.sample_rate * self.stride)
        k = k * (self.sample_rate * self.stride)

        start = int(i)
        end = int(start + 2048)
        item = self.samples_ds[0][start : end]
        print(" >> ", end - start) 
        return item 
"""
