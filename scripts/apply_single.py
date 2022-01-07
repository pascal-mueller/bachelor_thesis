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
    np.random.seed(80)
    torch.set_printoptions(precision=25)
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
