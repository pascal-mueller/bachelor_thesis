import torch
from torch import nn
import time
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from src.neuralNetwork import NeuralNetwork
from src.samplesDataset import SamplesDataset

from src.reg_BCELoss import reg_BCELoss

def train(TrainDL, network, loss_fn, optimizer):
    # Put model in train mode 
    network.train()
    
    loss = 0.0

    # TrainDL is an iterator. Each iteration gives us a bunch of
    # (samples, labels). The size of (samples, labels) depends on batch_size.
    # We put the TrainDL iterator in an enumerate to get the key/value pair. i_batch describes which batch we currently work through.
    # (samples, labels) is the actual data
    iterable = tqdm(TrainDL, desc=f"Training Network")

    for i_batch, (samples, labels) in enumerate(iterable):
        # Send data to device
        # [N, num_channels, length_data]
        samples = samples.unsqueeze(1)
        samples = samples.to(args.device)

        # Reset gradients TODO: Why?
        optimizer.zero_grad()

        # Compute prediction
        # samples[0] is a pytorch tensor of shape [1, 2048]
        labels_pred = network(samples.float())
        
        # Fix dimension
        labels_pred = labels_pred.squeeze(1).squeeze(1)

        # Get labels (cause use sigmoid)
        #labels = labels[:,0]

        # send to device
        labels = labels.to(args.device)

        loss_batch = loss_fn(labels_pred, labels.float())
        loss += loss_batch.item() # Get actual number

        # Backpropagation
        loss_batch.backward()

        # Clip gradients to make convergence somewhat easier # TODO: More research
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=100)

        # Make a step in the optimizer
        optimizer.step()
    
    return loss

def evaluation(ValidDL, network, loss_fn):
    # Put model into evaluation mode
    network.eval()

    # Variables
    loss = 0.0
    
    p_scores_USR = []
    p_scores_softmax = []
    labels_store = []

    correct = 0

    with torch.no_grad(): # TODO: Why use this?
        iterable = tqdm(ValidDL, desc=f"Validate trained Network")
        for i_batch, (samples, labels) in enumerate(iterable):
            # Send data to device
            # [N, num_channels, length_data]
            samples = samples.unsqueeze(1)
            samples = samples.to(args.device)

            # Get prediction
            labels_pred, labels_pred_USR = network(samples.float())

            # Fix dimension
            labels_pred = labels_pred.squeeze(1).squeeze(1)

            # Get labels (cause use sigmoid)
            #labels = labels[:,0]
            
            # send to device
            labels = labels.to(args.device)
            
            # Compute loss
            #loss_fn.weight = weights
            loss_batch = loss_fn(labels_pred, labels.float())
            loss += loss_batch.item()

            # Compute how many are above threshold 0.4 (arbritary)
            labels_pred = labels_pred.cpu()
            labels = labels.cpu()
            idx = labels_pred[:,0] > 0.4
            labels_pred[idx == True] = torch.tensor([1.0, 0.0])
            labels_pred[idx == False] = torch.tensor([0.0, 1.0])
            correct += sum(torch.eq(labels, labels_pred)[:,0])
            
            # Store scored for efficiency computation later on.
            p_scores_USR.extend(labels_pred_USR.cpu())
            p_scores_softmax.extend(labels_pred.cpu()[:,0])
            labels_store.extend(labels[:,0].cpu())

    efficiency_USR = compute_efficiency(p_scores_USR, labels_store)
    efficiency_softmax = compute_efficiency(p_scores_softmax, labels_store)

    return loss, correct, efficiency_USR, efficiency_softmax 

def compute_efficiency(p_scores, labels, FAP=0.0001):
    # 1. Sort p-scores, largest first.
    # 2. Choose the threshold.
    #    The threshold is given by the x-th largest where as we choose x to be:
    #      x = argmin_(x' in N) ( x'/N_noise - FAP)^2
    #    where N_noise = # total amount of noise samples.
    # 3. Compute efficiency by using the formula:
    #      efficiency = N_(signal > t) / N_s
    #    where N_(signal>t) are # of signals with p_score > t and N_s is total
    #    # of signals.
    p_scores = np.array(p_scores)
    labels = np.array(labels)

    N_total = len(labels)
    N_signal = sum(labels)
    N_noise = N_total - N_signal
    
    idx_signals = (labels == 1.0)
    idx_noise = (labels == 0.0)

    p_scores_sorted = np.sort(p_scores[idx_noise])

    # From 2. we get x = N_noise * FAP
    x = int(N_noise * FAP)
    
    # Get threshold
    t = p_scores_sorted[-x]

    N_signal_t = sum(p_scores[idx_signals] > t)

    efficiency = N_signal_t / N_signal

    return efficiency

if __name__=='__main__':
    # Get CLI parser
    parser = ArgumentParser(description = "CNN for GW-Signal analysis")

    # Define arguments
    parser.add_argument('weights_file', type=str,
        help = "File contianing weights.", default = None, nargs='?')
    parser.add_argument('--train', action='store_true',
        help="Train the network.")
    parser.add_argument('--device', type=str, default = 'cpu',
        help="Device to train on.")
    parser.add_argument('--samples_file', type=str, default='../samples.hdf5',
        help="Sample data to train network on. This is gonna be split into \
        training and evaluation set with a 80/20 split.")

    # Get arguments
    args = parser.parse_args()

    # Set options
    torch.manual_seed(421123)
    np.random.seed(111232)

    # Get model
    # TODO: Proper state saving
    network = NeuralNetwork().to(args.device)
    #network = get_network().to(args.device)
    
    if args.train == None and args.weights_file == None:
        print("--train set to False and no weights given.")
        print("Doing nothing.")
        quit()

    if args.weights_file != None:
        print("Using pretrained model.")
        print(f"Reading weights from {args.weights_file}")
        
        # Load state
        network.load_state_dict(torch.load(args.weights_file))

    if args.train == True:
        print("Training network...")
        # Parameters
        learning_rate = 0.0001
        beta1 = 0.9
        beta2 = 0.999
        betas = (beta1, beta2)
        eps = 1e-8
        batch_size = 32
        epochs = 200
        best_loss = 1.0e10 # Impossibly bad value
        
        # Read samples dataset
        samplesDS = SamplesDataset(args.samples_file, device=args.device)

        # Make a 80/20 split for training/eval data
        k = len(samplesDS)
        indices = np.arange(k)
        np.random.shuffle(indices)
        train_indices = np.arange(0, int(k * 0.8), dtype='int')
        validation_indices = np.arange(int(k * 0.2), k, dtype='int')
        TrainDS = torch.utils.data.Subset(samplesDS, train_indices)
        ValidDS = torch.utils.data.Subset(samplesDS, validation_indices)
        
        # Get Dataloaders
        TrainDL = torch.utils.data.DataLoader(TrainDS, batch_size=batch_size,
            pin_memory=True, shuffle=True)
        ValidDL = torch.utils.data.DataLoader(ValidDS, batch_size=batch_size,
            pin_memory=True, shuffle=True)
         
        n_train = len(TrainDL)
        n_valid = len(ValidDL)

        # Determine how many noise and noise+signal samples we have in the
        # validation set.
        n_valid_signal = sum(samplesDS[validation_indices][1][:,0])
        n_valid_noise = sum(samplesDS[validation_indices][1][:,1])
        n_valid_total = len(validation_indices)

        # Get loss function
        loss_fn = reg_BCELoss(dim=2)

        # Get optimizer # TODO: arguments randomyl chosen
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate,
                betas=betas, eps=eps)
        
        #optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        #optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
        
        # Epochs
        print("\n Epoch | Training Loss | Validation Loss | Accuracy | Efficiency USR")
        print("----------------------------------------------------------------------")
        
        training_losses = []
        evaluation_losses = []
        efficiencies_USR = []
        efficiencies_softmax = []
        fig, axs = plt.subplots(1, 2)

        for i, t in enumerate(range(epochs)):
            # Train the NN
            training_loss = train(TrainDL, network, loss_fn, optimizer)
            training_loss /= n_train
            training_losses.append(training_loss)

            # Validate on unseen data
            evaluation_loss, correct, efficiency_USR, efficiency_softmax = evaluation(ValidDL, network, loss_fn)
            evaluation_loss /= n_valid
            evaluation_losses.append(evaluation_loss)
            
            # Compute accuracy
            accuracy = correct / n_valid_total

            # Store efficiency
            efficiencies_USR.append(efficiency_USR)
            efficiencies_softmax.append(efficiency_softmax)

            # Print the losses
            info_string = "    %i  |   %.8f  |    %.8f   |  %.4f  | %.4f   "

            print(info_string % (t, training_loss, evaluation_loss, accuracy, efficiency_USR))
            
            # Store weights
            if evaluation_loss < best_loss:
                best_loss = evaluation_loss
                torch.save(network.state_dict(), f"../data/best_weights/{i}.pt")
                print(f"Best weights stored in ../data/best_weights/{i}.pt")
            if i % 5 == 0:
                # Plot losses
                axs[0].plot(range(len(training_losses)), training_losses, "-o")
                axs[0].plot(range(len(evaluation_losses)), evaluation_losses, "-o")
                
                # Plot efficiencies
                axs[1].plot(range(len(efficiencies_USR)), efficiencies_USR)
                axs[1].plot(range(len(efficiencies_softmax)), efficiencies_softmax)

                fig.savefig("losses_and_efficiencies.png")
        
                with open("efficiencies_softmax.csv", "w") as f_eff:
                   for item in efficiencies_softmax:
                       f_eff.write(str(item) + "\n")

        print(training_losses)
        print(evaluation_losses)

        print("Storing losses and efficiencies.")
        with open("training_losses.csv", "w") as f_train:
            for item in training_losses:
                f_train.write(str(item) + "\n")

        with open("validation_losses.csv", "w") as f_valid:
            for item in evaluation_losses:
                f_valid.write(str(item) + "\n")

        with open("efficiencies_USR.csv", "w") as f_eff:
            for item in efficiencies_USR:
                f_eff.write(str(item) + "\n")

        with open("efficiencies_softmax.csv", "w") as f_eff:
            for item in efficiencies_softmax:
                f_eff.write(str(item) + "\n")

        plt.plot(range(len(training_losses)), training_losses, "-o")
        plt.plot(range(len(evaluation_losses)), evaluation_losses, "-o")
        plt.savefig("losses.png")
        plt.show()
        print("Done with training!\n")
