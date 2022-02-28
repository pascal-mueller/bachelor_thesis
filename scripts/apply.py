#!/usr/bin/env python

from tqdm import tqdm
import numpy as np
import h5py
from argparse import ArgumentParser
import torch
import os

from src.neuralNetwork import NeuralNetwork
from src.testDataset import TestDataset

def get_triggers(network, TestDL, stride, trigger_threshold, device, filename,
        USR = False):
    print("USR =", USR)
    # Each item is a piece of data of size "stride".
    # To compute the total amount of items i.e. len(TestDS) we do:
    # len(TestDS) = ( duration - k )  / stride
    # whereas duration = N / sample_rate (seconds) and k is the amount of
    # datasets stored in the hdf5 file. Note: We have several datasets because
    # real ligo data is segmented.

    triggers = []
    times = []
    labels_preds_softmax = np.zeros((len(TestDL.dataset), 2))
    labels_preds_linear = np.zeros((len(TestDL.dataset), 1))
    times_store = np.zeros(len(TestDL.dataset))
    start = 0
    end = 0

    # Loop over data
    iterable = tqdm(TestDL, desc=f"Evaluating {len(TestDS)} samples.")
    with torch.no_grad():
        for i, (times, samples) in enumerate(iterable):
            samples = samples.unsqueeze(1)
            samples = samples.to(device=device)

            # label = [1.0, 0.0] # noise + signal
            # label = [0.0, 1.0] # pure noise
            labels_pred_softmax, labels_pred_linear = network(samples.float()) # Input dim: [N, num_channel, len_data]
            
            torch.set_printoptions(precision=3, sci_mode=False)

            # Get p-score that samples contain a signal
            labels_pred_linear = labels_pred_linear.cpu()
            labels_pred_softmax = labels_pred_softmax.squeeze(1).cpu()

            
            # Store predictions
            k = len(samples)
            end = start + k
            labels_preds_softmax[start : end] = labels_pred_softmax
            times_store[start : end] = times
            start += k

            if USR == False:
                signals_pred = labels_pred_softmax[:, 0]
            else:
                signals_pred = labels_pred_linear
                

            # Check which predictions are above the trigger_threshold
            trigger_bools = torch.gt(signals_pred, trigger_threshold).cpu()

            for j, trigger_bool in enumerate(trigger_bools):
                if trigger_bool == True:
                    time = times[j].item()
                    signal_pred = signals_pred[j].tolist()
                    triggers.append([time, signal_pred])

    print(f"A total of {len(triggers)}/{len(TestDS)} have exceeded the threshold of \
            {trigger_threshold}")
    
    # Store predictions
    print("Storing predicted labels.")

    with h5py.File(filename, 'w') as outfile:
        outfile.create_dataset('labels_pred_softmax', data=labels_preds_softmax)
        outfile.create_dataset('labels_pred_linear', data=labels_pred_linear)
        outfile.create_dataset('times', data=times_store)
        print("Predicted labels stored.")

    return triggers

def get_clusters(triggers, cluster_threshold):
    n_triggers = len(triggers)

    # Create iterator
    triggers = iter(triggers)
    
    # Read first trigger
    first_trigger = next(triggers) # Format: [time, signal_pred]
    old_trigger_time = first_trigger[0]
    
    # Frist trigger will of course start the first cluster
    clusters = [[first_trigger]]
    
    for trigger in triggers:
        new_trigger_time = trigger[0]
        time_diff = new_trigger_time - old_trigger_time
        
        # If current and last trigger are far enough apart, create a new cluster
        if time_diff >= cluster_threshold:
            clusters.append([trigger])
        # If they aren't far enough apart, add the current trigger to the
        # last cluster
        else:
            clusters[-1].append(trigger)
        
        old_trigger_time = new_trigger_time
    
    #new_clusters = []
    #for cluster in clusters:
    #    if len(cluster) > 3:
    #        new_clusters.append(cluster)

    #clusters = new_clusters


    print(f"Clustering has results in {len(clusters)}/{n_triggers} independent events.")
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

def read_triggers(filename, trigger_threshold=0.3, USR = False):
    print("USR =", USR)
    triggers = []

    with h5py.File(filename, 'r') as f:
        times = f["times"]

        if USR == False:
            labels_softmax = f["labels_pred_softmax"] 
            signals_pred = labels_softmax[:, 0]
        else:
            labels_linear  = f["labels_pred_linear"] 
            signals_pred =  labels_linear[:]

        #signals_pred = torch.tensor(labels_pred[:, 0])
        # Check which predictions are above the trigger_threshold
        trigger_bools = signals_pred >= trigger_threshold # element wise
        iterable = tqdm(trigger_bools, desc=f"Reading triggers from file.")
        for i, trigger_bool in enumerate(iterable):
            if trigger_bool == True:
                time = times[i]
                signal_pred = signals_pred[i].tolist()

                triggers.append([time, signal_pred])

    return triggers

if __name__=='__main__':
    # Get CLI parser
    parser = ArgumentParser(description = "CNN for GW-Signal analysis")

    # Define arguments
    parser.add_argument('trigger_threshold', type=float,
        help = "Trigger threshold.", default = 0.2, nargs='?')

    parser.add_argument('cluster_threshold', type=float,
        help = "Cluster threshold.", default = 0.3, nargs='?')

    parser.add_argument('weights_file', type=str,
        help = "File contianing weights.", default = None, nargs='?')

    parser.add_argument('inputfile', type=str,
        help = "File contianing samples.", default = None, nargs='?')

    parser.add_argument('output_file', type=str,
        help = "Filename of file contianing output.", default = None, nargs='?')

    parser.add_argument('predictions_file', type=str,
        help = "File containing predicted labels.", default = None, nargs='?')

    parser.add_argument('--device', type=str, default = 'cuda',
        help="Device to train on.")

    # Get arguments
    args = parser.parse_args()
    
    # Create dataset and dataloader
    TestDS = TestDataset(args.inputfile, device=args.device)
    TestDL = torch.utils.data.DataLoader(TestDS, batch_size=16384) # 8x1s
    
    # Get trained network
    # TODO: We need to load/save more than just the state
    network = NeuralNetwork().to(args.device)
    network.load_state_dict(torch.load('best_weights.pt', map_location=args.device))
    network.eval()
    
    store = os.path.isfile(args.predictions_file)

    # Use USR?
    USR = False

    if store == True:
        print("\nReading triggers from file: ", args.predictions_file)
        triggers = read_triggers(args.predictions_file, USR=USR)
    else:
        print("\nGetting triggers by evaluating test data.")
        triggers = get_triggers(network,
                                TestDL,
                                stride=0.1,
                                trigger_threshold=args.trigger_threshold,
                                device=args.device,
                                filename=args.predictions_file,
                                USR=USR)

    print(triggers[0:4])
    print("") 
    #print(f"Test Dataset has {len(TestDS)} items.")
    #print("Found ", len(triggers), " above threshold ", args.trigger_threshold)

    time, stat, var = get_clusters(triggers, args.cluster_threshold)

    #print("Categorized ", len(triggers), "triggers into ", len(time), "clusters with threshold", args.cluster_threshold)

    print("\nStatistics:")
    print("Trigger Threshold: ", args.trigger_threshold)
    print("Cluster Threshold: ", args.cluster_threshold)
    print("")
    print("Total items:    ", len(TestDS))
    print("Total Triggers: ", len(triggers))
    print("Total Clusters: ", len(stat))
    print("")

    print("Saving clustered triggers into %s." % args.output_file)
    with h5py.File(args.output_file, 'w') as outfile:
        ### Save clustered values to the output file and close it
        outfile.create_dataset('time', data=time)
        outfile.create_dataset('stat', data=stat)
        outfile.create_dataset('var', data=var)

        print("Triggers saved, closing file.\n")
