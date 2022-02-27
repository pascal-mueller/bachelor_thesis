import matplotlib.pyplot as plt
import h5py

from src.signalSpace import SignalSpace
from src.signalGenerator import SignalGenerator


plt.plot([1,2,3],[1,2,3])
plt.show()
quit()

signal_space = SignalSpace(1, 1)
signal_gen = SignalGenerator()
signal_params = next(signal_space)

signal = signal_gen.generate(signal_params)

t = range(len(signal))
plt.plot(t, signal)
plt.show()
