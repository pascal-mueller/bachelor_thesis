from torch import nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, USR = False):
        super().__init__()
        # TODO: Split network into features() and classifier() using
        # sequential module. This currently is a bit ugly.
        #self.network = get_network(path_to_weights)
        
        # Input Layer
        self.normalizeInput = nn.BatchNorm1d(1)
        
        # 1. Layer
        self.conv1 = nn.Conv1d(1, 8, 64)
        self.ELU1 = nn.ELU()

        # 2. Layer
        self.conv2 = nn.Conv1d(8, 8, 32)
        self.maxpool1 = nn.MaxPool1d(4)
        self.ELU2 = nn.ELU()

        # 3. Layer
        self.conv3 = nn.Conv1d(8, 16, 32)
        self.ELU3 = nn.ELU()

        # 4. Layer
        self.conv4 = nn.Conv1d(16, 16, 16)
        self.maxpool2 = nn.MaxPool1d(3)
        self.ELU4 = nn.ELU()
        
        # 5. Layer
        self.conv5 = nn.Conv1d(16, 32, 16)
        self.ELU5 = nn.ELU()

        # 6. Layer
        self.conv6 = nn.Conv1d(32, 32, 16)
        self.maxpool3 = nn.MaxPool1d(2)
        self.ELU6 = nn.ELU()
        
        # 7. Layer
        self.flatten = nn.Flatten()
        
        # 8. Layer
        self.linear1 = nn.Linear(1856, 64)
        self.dropout1 = nn.Dropout(p=0.8) #TODO: Decide p
        self.ELU7 = nn.ELU()
        
        # 9. Layer
        self.linear2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.8) #TODO: Decide p
        self.ELU8 = nn.ELU()

        self.linear3 = nn.Linear(64, 2)
        
        self.softmax = nn.Softmax(dim=1)
        self.USR = lambda z : z[:,0] - z[:,1]

    def forward(self, inputs):
        #z = self.network(inputs)

        # Input layer
        z = self.normalizeInput(inputs)

        # 1. Layer - Conv1D + ELU
        z = self.conv1(z)
        z = self.ELU1(z)
        
        # 2. Layer - Conv1Da + MaxPool1D + ELU
        z = self.conv2(z)
        z = self.maxpool1(z)
        z = self.ELU2(z)

        # 3. Layer - Conv1D + ELU
        z = self.conv3(z)
        z = self.ELU3(z)

        # 4. Layer - Conv1d + MaxPool1D + ELU
        z = self.conv4(z)
        z = self.maxpool2(z)
        z = self.ELU4(z)

        # 5. Layer - Conv1D + ELU
        z = self.conv5(z)
        z = self.ELU5(z)

        # 6. Layer - Conv1D + MaxPool1D + ELU
        z = self.conv6(z)
        z = self.maxpool3(z)
        z = self.ELU6(z)

        # 7. Layer - Flatten
        z = self.flatten(z)

        # 8. Layer - Linear + Dropout + ELU
        z = self.linear1(z)
        z = self.dropout1(z)
        z = self.ELU7(z)

        # 9. Layer - Linear + Dropout + ELU
        z = self.linear2(z)
        z = self.dropout2(z)
        z = self.ELU8(z)

        # Output Layer - Linear + Softmax
        logits = self.linear3(z)
        z_bounded = self.softmax(logits)
        
        # If we evaluate network, also return logits passed to USR instead of
        # Softmax. used for efficiency. Can't use same for loss & accuracy since
        # USR changes the labels.
        if self.training == True:
            return z_bounded.unsqueeze(1)
        else:
            z_unbounded = self.USR(logits)
            
            return z_bounded.unsqueeze(1), z_unbounded


"""
def get_network():
    Network = nn.Sequential(
        nn.BatchNorm1d(1),	# 1x2048
        nn.Conv1d(1, 8, 64),	# 8x1985
        nn.ELU(),
        nn.Conv1d(8, 8, 32),	# 8x1954
        nn.MaxPool1d(4),	# 8x488
        nn.ELU(),
        nn.Conv1d(8, 16, 32),	# 16x457
        nn.ELU(),
        nn.Conv1d(16, 16, 16),	# 16x442
        nn.MaxPool1d(3),	# 16x147
        nn.ELU(),
        nn.Conv1d(16, 32, 16),	# 32x132
        nn.ELU(),
        nn.Conv1d(32, 32, 16),	# 32x117
        nn.MaxPool1d(2),	# 32x58
        nn.ELU(),
        nn.Flatten(),	#  1856
        nn.Linear(1856, 64),	# 64
        nn.Dropout(p=.5),
        nn.ELU(),
        nn.Linear(64, 64),	# 64
        nn.Dropout(p=.5),
        nn.ELU(),
        nn.Linear(64, 2),	# 2
        nn.Softmax(dim=1)
    )

    return Network
"""
