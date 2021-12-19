import torch
from argparse import ArgumentParser

from src.neuralNetwork import NeuralNetwork
from src.testDataset import TestDataset



if __name__=='__main__':
    # Get CLI parser
    parser = ArgumentParser(description = "CNN for GW-Signal analysis")

    # Define arguments
    parser.add_argument('weights_file', type=str,
        help = "File contianing weights.", default = None, nargs='?')

    # Get arguments
    args = parser.parse_args()
    
    testfile = "../data/test_data/foreground_file.hdf"

    TestDS = TestDataset(testfile, device='cpu')
    TestDL = torch.utils.data.DataLoader(TestDS, batch_size=256)
    
    # Get trained network
    # TODO: We need to load/save more than just the state
    network = NeuralNetwork().to(device='cpu')
    network.load_state_dict(torch.load(args.weights_file, map_location=torch.device('cpu')))

    # Loop over data
    labels_pred = []
    for (i, item) in enumerate(TestDL):
        print(item)
        predictions = network(item.float())
        labels_pred.extend(predictions)

    print(labels_pred)
        

