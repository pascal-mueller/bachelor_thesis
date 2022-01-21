import h5py
import pycbc.detector, pycbc.waveform, pycbc.types
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.signalSpace import SignalSpace
from src.signalGenerator import SignalGenerator
from src.noiseGenerator import NoiseGenerator 

class SingleTestDataset(torch.utils.data.Dataset):
    def __init__(self, psd, stride=0.1, device='cpu'):
        self.sample_rate = 2048
        self.psd = psd
        self.stride = stride
        self.rng = np.random.default_rng()

        N = 5

        self.target_snrs = np.linspace(5.0, 30.0, N)
        self.signals = self.get_signals(N)
        self.duration = int(len(self.signals) / 2048)
        self.noise = self.get_noise()
        self.strain = self.get_strain()

    def get_noise(self):
        print("GET NOISE: ", self.duration*self.sample_rate)
        noise_gen = NoiseGenerator(self.psd, duration = self.duration)

        return noise_gen.generate() 

    
    def get_signals(self, N):
        signal_space = SignalSpace(N=N, stride=1)
        signal_gen = SignalGenerator()

        signals = [0] * (self.sample_rate * 4) # 4 seconds

        padding = 3 # seconds
        for i, signal_params in enumerate(signal_space):
            signal = signal_gen.generate(signal_params)[0]

            signal = self.SNR_scale(signal, self.target_snrs[i])
            
            k = int(np.ceil(len(signal) / self.sample_rate))
            
            # Allocate temporarily memory that holds the signal and 3 seconds
            # of padding behind it.
            tmp = [0] * ((k + 3) * self.sample_rate)
            
            tmp[0 : len(signal)] = signal
            # Add signal
            signals.extend(tmp)
        
        # Add 1 second of zeros at the end. That's because the last signal is the one which the biggest SNR and somehow we get an artifact
        # (high signal probability) at the very end of the plot. That's probably
        # due to whitening but I'm not 100% sure. Adding 1s of zeros solves it.
        tmp = [0] * (1*self.sample_rate)
        signals.extend(tmp)

        return signals
    
    def get_strain(self):
        n_noise = len(self.noise)
        n_signal = len(self.signals)

        k_inj = int(0.5 * (n_noise - n_signal))

        strain = self.noise.copy()

        # Scale SNR and also make injection
        print(len(self.noise), len(self.signals))
        strain = self.noise + self.signals # Actual injection
        strain = self.whiten(strain) 
        
        return strain

    def SNR_scale(self, signal, target_snr):
        signal = pycbc.types.TimeSeries(signal, 
                             delta_t = 1.0 / self.sample_rate, dtype=np.float64)
        foo = pycbc.filter.matchedfilter.sigmasq(signal, psd=self.psd,
                low_frequency_cutoff = 18.0)

        network_snr = np.sqrt(foo)
        print("target = ", target_snr)
        
        return signal * (target_snr / network_snr)

        # TODO: Understand snr scaling here
        #self.signals = signal.numpy() * (target_snr/network_snr)
        #sample = self.noise + self.signals
        """
        fix, axs = plt.subplots(4, 1)
        axs[0].plot(range(len(self.noise)), self.noise)
        axs[1].plot(range(len(signal)), signal)
        axs[2].plot(range(len(self.noise)), self.noise)
        axs[2].plot(range(len(signal)), signal)
        axs[3].plot(range(len(sample)), sample)
        plt.show()
        """
        #return sample
    
    def whiten(self, strain):
        # Whiten
        strain = pycbc.types.TimeSeries(strain, delta_t = 1.0 / self.sample_rate)
        # TODO: How coose params for whiten?
        # TODO: After whitening we only have 1s left. Input was 1.5s.
        # How do we get exaclty 1s?
        # ASSUMING 1.25 s
        k = int(self.sample_rate * 0.125)
        strain.prepend_zeros(k)
        strain.append_zeros(k)
        strain = strain.whiten(0.5, 0.25, remove_corrupted = True,
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
        axs[0].set_xticks(np.arange(0,len(t)+1, 2048))
        axs[0].set_xticklabels(np.arange(0,len(t)+1, 2048), rotation=45)

        # Plot strain (noise + signal)
        axs[1].plot(t, self.strain)
        axs[1].set_title("SNR scaled & whitened strain (Noise + signal)")
        axs[0].set_xticks(np.arange(0,len(t)+1, 2048))
        axs[0].set_xticklabels(np.arange(0,len(t)+1, 2048), rotation=45)

        if len(probabilities) > 0:
            axs[2].plot(t, probabilities)
            axs[2].set_title("Probability signal")
            axs[2].set_xticks(np.arange(0,len(t)+1, 2048))
            axs[2].set_xticklabels(np.arange(0,len(t)+1, 2048), rotation=45)
        
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
