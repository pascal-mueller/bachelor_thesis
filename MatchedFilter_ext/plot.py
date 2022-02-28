import matplotlib.pyplot as plt
import numpy as np
import h5py

#k = 30 * 24 * 60 * 60

with h5py.File('statistics.hdf', 'r') as f:
    far = f['far'][:]
    far = far
    sens_dist = f['sens-dist'][:]
    
    fig, ax = plt.subplots()

    ax.semilogx(far, sens_dist)

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmax, xmin)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)

    plt.show()
    plt.savefig("matchedfiltering.png")
