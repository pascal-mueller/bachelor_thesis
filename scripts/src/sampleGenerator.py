import pycbc.detector, pycbc.waveform, pycbc.types
import numpy as np

class SampleGenerator:
    def __init__(self, file):
        self.sample_rate = 2048
        self.duration = 1.25
        self.rng = np.random.default_rng()
        self.file = file
        self.noise_ds = file['noise']
        self.signals_ds = file['signals']
        self.samples_ds = file['samples']
        self.labels_ds = file['samples_labels']

        self.snr = []

    def generate(self, params):
        samples = np.zeros((len(params), 2048))
        labels = np.zeros(len(params))

        for (i, param) in enumerate(params):
            index = param['index']
            idx_noise = param['idx_noise']
            idx_signal = param['idx_signal']
            
            # Pure noise
            if idx_signal == None:
                noise = self.noise_ds[idx_noise]
                # Given: len(noise) = 2560 i.e. 1.25s
                # Need: 1s
                # => We cut 1s out of the 1.25s around the middle.
                
                # Cut out duration*sample_rate around the middle
                k_half = int(0.5 * 1 * self.sample_rate)
                l = int(0.5 * len(noise))
                noise = noise[l - k_half : l + k_half] 
                
                # Write into dataset
                #self.samples_ds[index] = noise
                samples[i] = noise
                labels[i] = 0
                #self.labels_ds[index] = 0
            
            # Noise + Signal
            else:
                noise = self.noise_ds[idx_noise]
                signal = self.signals_ds[idx_signal]
                

                noise = pycbc.types.TimeSeries(noise,
                        delta_t = 1.0 / self.sample_rate, dtype=np.float64)
                signal = pycbc.types.TimeSeries(signal,
                        delta_t = 1.0 / self.sample_rate, dtype=np.float64)
                              

                # Scale SNR and inject
                sample = self.SNR_scale(signal, noise)

                # Remember: Whiten turns 1.25s into 1s.
                sample = self.whiten(sample)

                # Store
                #self.samples_ds[index] = sample
                #self.labels_ds[index] = 1
                samples[i] = sample
                labels[i] = 1

        idx = params[0]['index']
        k = len(params)
        
        """
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(params), 3)
        fig.suptitle("hi")
        k = 0

        para = params.copy()
        for (i, param) in enumerate(para):
            index = param['index']
            idx_noise = param['idx_noise']
            idx_signal = param['idx_signal']
            
            if idx_signal != None:
                signal = self.signals_ds[idx_signal]
                k = k + 1
                print("scaled with ", self.snr[k-1])
            else:
                signal = np.zeros(2560)

            noise = self.noise_ds[idx_noise]
            
            t = range(len(noise))
            axs[i][0].plot(t, noise)
            axs[i][1].plot(t, signal*self.snr[k-1])
            axs[i][2].plot(range(len(samples[i])), samples[i])

        import time
        plt.savefig(f"plot-{time.time()}.png")
        """

        self.samples_ds[idx : idx + k] = samples
        self.labels_ds[idx : idx + k] = labels

    def SNR_scale(self, signal, noise):
        # Create PSD for detector
        psd_length = int(0.5 * self.sample_rate * self.duration) + 1
        delta_f = 1.0 / self.duration
        psd_fn = pycbc.psd.analytical.aLIGOZeroDetHighPower
        psd = psd_fn(length=psd_length, delta_f=delta_f, low_freq_cutoff=18.0)
    
        # SNR scaling
        foo = pycbc.filter.matchedfilter.sigmasq(signal, psd=psd,
                low_frequency_cutoff = 18.0)
        network_snr = np.sqrt(foo)
        target_snr = self.rng.uniform(5.0, 15.0)

        #print("network_snr=", network_snr, " target_snr=", target_snr, "ratio=", target_snr/network_snr)
        
        # TODO: Understand snr scaling here
        sample = noise + signal.numpy() * (target_snr/network_snr)

        self.snr.append(target_snr/network_snr)
    
        return sample
    
    def whiten(self, sample):
        # Whiten
        sample = pycbc.types.TimeSeries(sample, delta_t = 1.0 / self.sample_rate)
        # TODO: How coose params for whiten?
        # TODO: After whitening we only have 1s left. Input was 1.5s.
        # How do we get exaclty 1s?
        # ASSUMING 1.25 s
        sample = sample.whiten(0.5, 0.25, remove_corrupted = True,
                low_frequency_cutoff = 18.0)
        sample = sample.numpy()
    
        return sample
