from mpi4py import MPI # Will auto Init() and Finalize() MPI.
import pycbc.psd

from src.signalSpace import SignalSpace
from src.noiseSpace import NoiseSpace
from src.sampleSpace import SampleSpace
from src.signalGenerator import SignalGenerator
from src.noiseGenerator import NoiseGenerator
from src.sampleGenerator import SampleGenerator
from src.fileManager import FileManager

import time
from tqdm import tqdm
import sys

if __name__=='__main__':
    start_time = time.time()

    total_time = 0    
    worktime_signals = 0
    worktime_noise = 0
    worktime_samples = 0

    # Get MPI world
    world = MPI.COMM_WORLD
    world_rank = MPI.COMM_WORLD.rank
    
    # Create slaves (will access file)
    color = int(world_rank > 0)
    slaves = world.Split(color=color, key=world_rank)
   
    # Filename 
    filename = sys.argv[5]
    
    N_noise = int(sys.argv[1])
    N_signal = int(sys.argv[2])
    N_samples = int(sys.argv[3])

    psd = None
    if world_rank == 0:
        # Generate PSD
        # TODO: Why choose PSD length like that? Shouldn't I choose
        # int(0.5 * 2048 * 1.25) + 1?
        psd_length = int(2048 * 1.25)
        delta_f = 1.0 / 1.25
        psd_fn = pycbc.psd.aLIGOZeroDetHighPower
        psd = psd_fn(length=psd_length, delta_f=delta_f, low_freq_cutoff=15.0)

    # Send psd to all slaves
    psd = world.bcast(psd, root=0)
    print("Broadcasted PSD:", psd)

    # 0 Rank is our master, rest are slaves
    if world_rank == 0:
        stride = int(sys.argv[4])

        #
        # Generate signals
        #
        signal_space = SignalSpace(N_signal, stride)
        iterable = tqdm(signal_space, desc=f"Generating {N_signal} signal  samples")
        for (i, signal_params) in enumerate(iterable):
            # Receive a request from a slave to get more work
            dest = world.recv(source=MPI.ANY_SOURCE, tag=0)

            # Send the slave some work
            world.send(obj=signal_params, dest=dest, tag=dest)

        # Tell all the slaves: All work is done.
        for i in range(1, world.size):
            dest = world.recv(source=i, tag=0)
            world.send(obj=None, dest=dest, tag=dest)

        #
        # Generate noise
        #

        # NOTE: We create N_noise pure noise samples but also need N_signal
        # noise samples for all the signals.
        noise_space = NoiseSpace(N_noise + N_signal, stride)
        iterable = tqdm(noise_space, desc=f"Generating {N_noise + N_signal} noise samples")
        for (i, noise_params) in enumerate(iterable):
            # Receive a request from a slave to get more work
            dest = world.recv(source=MPI.ANY_SOURCE, tag=0)

            # Send the slave some work
            world.send(obj=noise_params, dest=dest, tag=dest)

        # Tell all the slaves: All work is done.
        for i in range(1, world.size):
            dest = world.recv(source=i, tag=0)
            world.send(obj=None, dest=dest, tag=dest)

        #
        # Generate samples
        #
        sample_space = SampleSpace(N_noise, N_signal, stride)
        iterable = tqdm(sample_space, desc=f"Generating {N_samples + N_noise} samples")
        for (i, sample_params) in enumerate(iterable):
            # Receive a request from a slave to get more work
            dest = world.recv(source=MPI.ANY_SOURCE, tag=0)

            # Send the slave some work
            world.send(obj=sample_params, dest=dest, tag=dest)

        # Tell all the slaves: All work is done.
        for i in range(1, world.size):
            dest = world.recv(source=i, tag=0)
            world.send(obj=None, dest=dest, tag=dest)
    # Slaves
    else:
        with FileManager(filename, N_noise, N_signal, comm=slaves) as file:
            #
            # Generate Signals
            #
            start = time.time()

            # Ask master for initial work.
            world.send(obj=world_rank, dest=0, tag=0)
            signal_gen = SignalGenerator(file)
            while(True):
                # Receive work from master
                signal_params = world.recv(source=0, tag=world_rank)
                if signal_params is None:
                    break
                
                # Do the actual work.
                signal_gen.generate(signal_params)

                # Ask the master for more work.
                world.send(obj=world_rank, dest=0, tag=0)

            worktime_signals = time.time() - start
            
            #
            # Generate Noise
            #
            start = time.time()
            
            # Ask master for initial work.
            world.send(obj=world_rank, dest=0, tag=0)
            noise_gen = NoiseGenerator(psd, file)
            while(True):
                # Receive work from master
                noise_params = world.recv(source=0, tag=world_rank)
                
                if noise_params is None:
                    break
                
                # Do the actual work.
                noise_gen.generate(noise_params)

                # Ask the master for more work.
                world.send(obj=world_rank, dest=0, tag=0)

            worktime_noise = time.time() - start

            #
            # Generate samples (noise + signal or pure noise)
            #

            start = time.time()

            world.send(obj=world_rank, dest=0, tag=0)
            sample_gen = SampleGenerator(psd, file)
            while(True):
                # Receive work from master
                sample_params = world.recv(source=0, tag=world_rank)

                if sample_params is None:
                    break
                
                # Do the actual work.
                sample_gen.generate(sample_params)

                # Ask the master for more work.
                world.send(obj=world_rank, dest=0, tag=0)
            worktime_samples = time.time() - start
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Rank {world_rank}: total={total_time}, signals={worktime_signals}, \
            noise={worktime_noise}, samples={worktime_samples}")
    
    """
    world.barrier()

    if world_rank == 0:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        import h5py
        
        file = h5py.File(filename, 'r')
        signals = file["samples"]
        
        fig, ax = plt.subplots()
        
        def animate(i):
            x = range(len(signals[i]))
            y = signals[i]

            ax.clear()
            ax.plot(x, y)
            ax.set_title(f"Sample Nr. {i}")
        
        anim = FuncAnimation(fig, animate, frames=200, interval=100, repeat=False)

        f = "samples.gif"
        writergif = PillowWriter(fps=30) 
        anim.save(f, writer=writergif)
        
        plt.show()
    """
        
"""
    1. Create MPI stuff
    2. Create FileWrapper object => creates datasets and all that
    3. Generate Noise using NoiseGenerator => Write to file
    4. Generate Signals using SignalGenerator => Write to file
    
    5. Now we want to draw X amount of noise+signal samples from Noise and
       Signals and SNR scale it and whiten it and write it to the file.

    6. Shuffle all Noise into the X amount of noise+signal

    Now for 5. and 6. needs the indices already. Master could use a Sampler
    to get those. We could the make the sampler into an iterator and iterate
    over all samples. Each slave gets a bunch of those samples and does the
    work.

    So basically we have three tasks:

    1st: Generate Noise
    2nd: Generate Signals
    3th: Inject and Shuffle

    For the sake of simplicity, we just put a barrier between each three. Each
    piece of work inside those subtasks is expected to take about the same
    amount of time, so it doesn't matter.
"""
