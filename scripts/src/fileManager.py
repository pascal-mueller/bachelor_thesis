import h5py

class FileManager:
    def __init__(self, filename, comm, N_noise, N_signal, N_samples):
        self.comm = comm
        self.file = h5py.File(filename, 'w', driver='mpio', comm=comm)
        
        duration = 1.25
        sample_rate = 2048
        length = duration * sample_rate

        self.file.create_dataset("noise", (N_noise, length), dtype='f')
        self.file.create_dataset("signals", (N_signal, length), dtype='f')
        self.file.create_dataset("samples", (N_samples + N_noise, sample_rate), dtype='f')
        self.file.create_dataset("samples_labels", (N_samples + N_noise,), dtype='i')
    
    def __def__(self):
        self.file.close() #TODO: Do I need this rly?

    def __enter__(self):
        return self.file

    def __exit__(self, type, value, traceback):
        self.file.close()
