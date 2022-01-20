import h5py

class FileManager:
    def __init__(self, filename, N_noise, N_signal, comm=None):
        if comm == None:
            self.file = h5py.File(filename, 'w')
        else:
            self.file = h5py.File(filename, 'w', driver='mpio', comm=comm)

        self.comm = comm
        
        duration = 1.25
        sample_rate = 2048
        length = duration * sample_rate

        self.file.create_dataset("noise", (N_noise + N_signal, length), dtype='d')
        self.file.create_dataset("signals", (N_signal, length), dtype='d')
        self.file.create_dataset("samples", (N_noise + N_signal, sample_rate), dtype='d')
        self.file.create_dataset("samples_labels", (N_noise + N_signal, ), dtype='i')

    def __enter__(self):
        return self.file

    def __exit__(self, type, value, traceback):
        self.file.close()
