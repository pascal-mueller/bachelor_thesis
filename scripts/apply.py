#!/usr/bin/env python

from tqdm import tqdm
import numpy as np
import h5py
from argparse import ArgumentParser
import torch

from src.neuralNetwork import NeuralNetwork
from src.testDataset import TestDataset

def get_triggers(network, TestDL, stride, trigger_threshold, device):
    # Each item is a piece of data of size "stride".
    # To compute the total amount of items i.e. len(TestDS) we do:
    # len(TestDS) = ( duration - k )  / stride
    # whereas duration = N / sample_rate (seconds) and k is the amount of
    # datasets stored in the hdf5 file. Note: We have several datasets because
    # real ligo data is segmented.

    triggers = []
    # Loop over data
    iterable = tqdm(TestDL, desc=f"Evaluating {len(TestDS)} samples.")
    with torch.no_grad():
        for times, samples in iterable:
            samples = samples.unsqueeze(1)
            samples = samples.to(device=device)

            # label = [1.0, 0.0] # noise + signal
            # label = [0.0, 1.0] # pure noise
            labels_pred = network(samples.float()) # Input dim: [N, num_channel, len_data]
    
            # Get p-score that samples contain a signal
            labels_pred = labels_pred.squeeze(1).cpu()
            signals_pred = labels_pred[:, 0] 

            # Check which predictions are above the trigger_threshold
            trigger_bools = torch.gt(signals_pred, trigger_threshold).cpu()
            
            for i, trigger_bool in enumerate(trigger_bools):
                if trigger_bool == True:
                    time = times[i].item()
                    signal_pred = signals_pred[i].tolist()

                    triggers.append([time, signal_pred])

    print(f"A total of {len(triggers)}/{len(TestDS)} have exceeded the threshold of \
            {trigger_threshold}")
    
    return triggers

def get_clusters(triggers, cluster_threshold):
    n_triggers = len(triggers)

    # Create iterator
    triggers = iter(triggers)
    
    #
    first_trigger = next(triggers)
    old_trigger_time = first_trigger[0]

    clusters = [[first_trigger]]
    

    for trigger in triggers:
        new_trigger_time = trigger[0]
        time_diff = new_trigger_time - old_trigger_time
        
        # If current and last trigger are far enough apart, create a new cluster
        if time_diff > cluster_threshold:
            clusters.append([trigger])
        # If they aren't far enough apart, add the current trigger to the
        # last cluster
        else:
            clusters[-1].append(trigger)
        
        old_trigger_time = new_trigger_time
    

    print(f"Clustering has results in {len(clusters)}/{n_triggers} independent triggers.")
    print("Centering triggers at their maxima.")

    cluster_times = []
    cluster_values = []
    cluster_timevars = [] # TODO: What is this?

    # Determine maxima of clusters and the corresponding times and append them
    # to the cluster_* lists
    for cluster in clusters:
        times = []
        values = []
        for trigger in cluster:
            times.append(trigger[0])
            values.append(trigger[1])

        max_index = np.argmax(values)
        cluster_times.append(times[max_index])
        cluster_values.append(values[max_index])
        cluster_timevars.append(0.2)

    cluster_times = np.array(cluster_times)
    cluster_values = np.array(cluster_values)
    cluster_timevars = np.array(cluster_timevars)

    return cluster_times, cluster_values, cluster_timevars

if __name__=='__main__':
    # Get CLI parser
    parser = ArgumentParser(description = "CNN for GW-Signal analysis")

    # Define arguments
    parser.add_argument('weights_file', type=str,
        help = "File contianing weights.", default = None, nargs='?')
    parser.add_argument('inputfile', type=str,
        help = "File contianing samples.", default = None, nargs='?')
    parser.add_argument('outputfile', type=str,
        help = "File contianing output.", default = None, nargs='?')
    parser.add_argument('--device', type=str, default = 'cuda',
        help="Device to train on.")

    # Get arguments
    args = parser.parse_args()
    
    # Create dataset and dataloader
    TestDS = TestDataset(args.inputfile, device=args.device)
    TestDL = torch.utils.data.DataLoader(TestDS, batch_size=16384)

    # Get trained network
    # TODO: We need to load/save more than just the state
    network = NeuralNetwork().to(args.device)
    network.load_state_dict(torch.load(args.weights_file))

    triggers = get_triggers(network,
                            TestDL,
                            stride=0.1,
                            trigger_threshold=0.505,
                            device=args.device)

    cluster_threshold = 0.3
    time, stat, var = get_clusters(triggers, cluster_threshold)

    print("Saving clustered triggers into %s." % args.outputfile)
    print(args.outputfile, type(args.outputfile))
    with h5py.File(args.outputfile, 'w') as outfile:
        ### Save clustered values to the output file and close it
        outfile.create_dataset('time', data=time)
        outfile.create_dataset('stat', data=stat)
        outfile.create_dataset('var', data=var)

        print("Triggers saved, closing file.")
