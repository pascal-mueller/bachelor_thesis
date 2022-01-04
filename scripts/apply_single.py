import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.signalSpace import SignalSpace
from src.noiseSpace import NoiseSpace
from src.sampleSpace import SampleSpace
from src.signalGenerator import SignalGenerator
from src.noiseGenerator import NoiseGenerator
from src.sampleGenerator import SampleGenerator
from src.fileManager import FileManager

from src.neuralNetwork import NeuralNetwork
from src.testDataset import TestDataset
from src.samplesDataset import SamplesDataset
from src.singleTestDataset import SingleTestDataset 

if __name__=='__main__':
    torch.set_printoptions(precision=25)
    """
    filename = '../data/test_data_single/test_data_single.hdf5'
    N_noise = 10
    N_signal = 10
    N_samples = 1
    stride = 1
    with FileManager(filename, N_noise, N_signal, N_samples) as file:
        # Generate signal
        signal_space = SignalSpace(N_signal, stride)
        signal_gen = SignalGenerator(file)
        signal_params = next(signal_space)
        signal_gen.generate(signal_params)
        
        # Generate noise
        noise_space = NoiseSpace(N_noise, stride)
        noise_gen = NoiseGenerator(file)
        noise_params = next(noise_space)
        noise_gen.generate(noise_params)

        # Generate sample
        sample_space = SampleSpace(N_noise, N_signal, N_samples, stride)
        sample_gen = SampleGenerator(file)
        for sample_param in sample_space:
            sample_gen.generate(sample_param)
    """
    duration = 10
    sample_rate = 2048
    device = 'cpu'
    probabilities = np.zeros(sample_rate * duration)

    # Create dataset and dataloader
    TestDS = SingleTestDataset(stride=0.1, duration=duration, device=device)
    TestDL = torch.utils.data.DataLoader(TestDS, batch_size=10)

    # Apply data
    network = NeuralNetwork().to(device)
    network.load_state_dict(torch.load('best_weights.pt'))
    
    iterable = tqdm(TestDL, desc=f"Evaluating {duration} of test data.")
    with torch.no_grad():
        for start, end, strains in iterable:
            strains = strains.unsqueeze(1)
            
            # Dim: [1, 2048] or [N, 1, 2048] resp. [N, C, length]
            labels_pred = network(strains.float())
            labels_pred = labels_pred[:,0,0].tolist()

            for i, label in enumerate(labels_pred):
                probabilities[start[i] : end[i]] = label

    TestDS.plot(probabilities)

    print("Done")
