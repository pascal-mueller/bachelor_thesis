import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.neuralNetwork import NeuralNetwork
from src.singleTestDataset import SingleTestDataset 

if __name__=='__main__':
    np.random.seed(99999998)
    torch.manual_seed(34567822)
    torch.set_printoptions(precision=25)
    sample_rate = 2048
    device = 'cpu'

    # Create dataset and dataloader
    TestDS = SingleTestDataset(stride=0.1, device=device)
    TestDL = torch.utils.data.DataLoader(TestDS, batch_size=10)
    
    # Apply data
    network = NeuralNetwork().to(device)
    network.load_state_dict(torch.load('best_weights.pt', map_location=device))
    
    probabilities = np.zeros(int(len(TestDL) * sample_rate))
    print(probabilities[-2048:-1])

    iterable = tqdm(TestDL, desc=f"Evaluating simple test data.")
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
