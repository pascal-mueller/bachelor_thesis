import pycbc.psd, pycbc.noise
import numpy as np
import time

psd_length = int(0.5 * 2048 * 1.25) + 1
delta_f = 1.0 / 1.25
psd_fn = pycbc.psd.analytical.aLIGOZeroDetHighPower
psd = psd_fn(length=psd_length, delta_f=delta_f, low_freq_cutoff=15.0)

old_noise = None

for i in range(100000):
    if i % 10000 == 0:
        print(i)
    noise_fn = pycbc.noise.gaussian.frequency_noise_from_psd
    noise = noise_fn(psd).to_timeseries().numpy()

    if np.array_equal(old_noise, noise):
        print("SAME NOISE")

    old_noise = noise
