import torch
import numpy as np
from argparse import ArgumentParser

from src.neuralNetwork import NeuralNetwork
from src.samplesDataset import SamplesDataset

from src.reg_BCELoss import reg_BCELoss

def train(TrainDL, network, loss_fn, optimizer):
    # Put model in train mode
    network.train()
    
    loss = 0.0

    # TrainDL is an iterator. Each iteration gives us a bunch of
    # (samples, labels). The size of (samples, labels) depends on batch_size.
    # We put the TrainDL iterator in an enumerate to get the key/value pair.
    # i_batch describes which batch we currently work through.
    # (samples, labels) is the actual data
    for i_batch, (samples, labels) in enumerate(TrainDL):
        print(samples)
        quit()
        # Reset gradients TODO: Why?
        optimizer.zero_grad()

        # Compute prediction
        # samples[0] is a pytorch tensor of shape [1, 2048]
        labels_pred = network(samples.float())
        
        # Get weights used for given sample
        # TODO: Check weights
        weights = labels[:,0,0]
        weights[weights==1.0] = 0.1
        weights[weights==0.0] = 1.0
        
        loss_fn2 = reg_BCELoss(dim=2)

        # Compute loss
        loss_batch = loss_fn(labels_pred, labels)
        loss_batch2 = loss_fn2(labels_pred, labels)
        print(loss_batch)
        print(loss_batch2)
        quit()
        loss += loss_batch.item() # Get actual number

        # Backpropagation
        loss_batch.backward()

        # Clip gradients to make convergence somewhat easier # TODO: More research
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=100)

        # Make a step in the optimizer
        optimizer.step()
    
    return loss

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
    torch.manual_seed(42)

    # Get model
    # TODO: Proper state saving
    network = NeuralNetwork().to(args.device)
    
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
        learning_rate = 1e-5 
        beta1 = 0.9
        beta2 = 0.999
        betas = (beta1, beta2)
        eps = 1e-8
        batch_size = 3
        epochs = 20
        best_loss = 1.0e10 # Impossibly bad value
        
        # Read samples dataset
        samplesDS = SamplesDataset('../data/train_eval_set.hdf',
            device=args.device)

        # Make a 80/20 split for training/eval data
        k = len(samplesDS)
        train_indices = np.arange(0, int(k * 0.8), dtype='int')
        validation_indices = np.arange(int(k * 0.8), k, dtype='int')
        TrainDS = torch.utils.data.Subset(samplesDS, train_indices)
        ValidDS = torch.utils.data.Subset(samplesDS, validation_indices)
        
        # Get Dataloaders
        TrainDL = torch.utils.data.DataLoader(TrainDS, batch_size=batch_size,
            shuffle=False)
        ValidDL = torch.utils.data.DataLoader(ValidDS, batch_size=batch_size,
            shuffle=False)
        
        n_train = len(TrainDS)
        n_valid = len(ValidDS)

        # Get loss function
        loss_fn = reg_BCELoss(dim=2, reduction = 'none')

        # Get optimizer # TODO: arguments randomyl chosen
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate,
                betas=betas, eps=eps)
        
        # Epochs
        print("\nEpoch  |  Training Loss  |  Evaluation Loss")
        print("-------------------------------------------")
        for t in range(epochs):
            # Train the NN
            training_loss = train(TrainDL, network, loss_fn, optimizer)
            training_loss /= n_train

            # Evaluate on unseen data
            evaluation_loss = evaluation(ValidDL, network, loss_fn)
            evaluation_loss /= n_valid
            
            # Print the losses
            info_string = "   %i   |      %.2f      |      %.2f" % (t, training_loss, evaluation_loss)
            print(info_string)
            
            # Store weights
            if evaluation_loss < best_loss:
                best_loss = evaluation_loss
                torch.save(network.state_dict(), 'best_weights.pt')
                print("Best weights stored in best_weights.pt")

        print("Done with training!\n")
