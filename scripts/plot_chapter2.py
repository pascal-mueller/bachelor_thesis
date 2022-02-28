import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import h5py
import pycbc

from src.signalSpace import SignalSpace
from src.signalGenerator import SignalGenerator

def get_signal(param):
    param = param[0]
    param['waveform_kwargs']['mass1'] = 25
    param['waveform_kwargs']['mass2'] = 20

    
    rng = np.random.default_rng()
    duration = 1.25
    sample_rate = 2048
    detector = pycbc.detector.Detector('H1')

    # Take the injection time randomly in the LIGO O3a era              
    injection_time = rng.uniform(1238166018, 1253977218)

    print("injection_time=", injection_time)

    # Generate the full waveform - VARIABLE LENGTH
    waveform_kwargs = param['waveform_kwargs']
    waveform = pycbc.waveform.get_td_waveform(**waveform_kwargs)
    hp, hc = waveform

    # Note: The start_time of the waveform might be negative because
    # time axis is moved s.t. t=0 is at the actual merger.
    
    # Note: hp.get_sample_times()[0] can be negative since t=0 is at
    # merger time.
    
    # Place merger at injection_time
    print("0 time=", hp.get_sample_times())
    start_time = injection_time + hp.get_sample_times()[0]
    print("start_time=", start_time)
    hp.start_time = start_time
    hc.start_time = start_time

    # The waveform generator stops the waveform soon after the merger.
    # So we append some zeros to be sure that we actually have enough
    # long of a waveform to slice it later.
    hp.append_zeros(duration * sample_rate)
    hc.append_zeros(duration * sample_rate)

    # Project generated waveform onto detector. Note that this
    # projection depends on the start_time. That is because earth is
    # moving.
    right_ascension = param['right_ascension']
    declination = param['declination']
    pol_angle = param['pol_angle']
    strain = detector.project_wave(hp, hc, right_ascension,
            declination, pol_angle)
    
    # The merger is where the maximal value is. We get the index of that
    # maximal value.
    # NOTE: This isn't theoretically true for ANY merger but it's close enough.
    # It could be, that the highest value is close around the merger. Since
    # we vary it in time anyway, it shouldn't matter too much. I also don't
    # expect it to occure very often.
    idx_merger = np.argmax(strain)

    # Cut 1.25s around merger.
    k = int(duration * sample_rate)
    k_half = int(0.5 * k)
    
    # Vary the merger by at most 0.2s seconds 
    k_vary = int(rng.uniform(-0.2, 0.2) * sample_rate)
    idx_merger += k_vary
    
    # Set start and end index for cutting window
    start = idx_merger - k_half
    end = idx_merger + k_half
    
    # If merger is close to left or right end of the strain, we might
    # not be able to center our cutting window around it but have to
    # respect the left or right border.
    if start < 0:
        start = 0 # Set start to left border
        end = start + k

    if end > len(strain):
        end = len(strain) # Set end to right border
        start = end - k

    return strain, start, end

signal_space = SignalSpace(1,1)
signal_gen = SignalGenerator()
signal_params = next(signal_space)

signal, start, end = get_signal(signal_params)


fig, ax = plt.subplots(1, figsize=(19,6))

ax.plot(range(len(signal)), signal)

ax.axvline(start-256, color='g')
ax.axvline(start, color='r')
ax.axvline(start+256, color='y')

ax.axvline(end-256, color='g')
ax.axvline(end, color='r')
ax.axvline(end+256, color='y')

ax.set_xlabel("Time [s]")
ax.set_ylabel("Waveform [-]")
x = np.arange(0, len(signal), 2048)
ax.set_xticks(x, x/2048)

plt.savefig("chapter2_cutting_window.png", bbox_inches="tight")
plt.show()

