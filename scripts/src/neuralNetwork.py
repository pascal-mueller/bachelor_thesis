from torch import nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Split network into features() and classifier() using
        # sequential module. This currently is a bit ugly.
        #self.network = get_network(path_to_weights)
        self.normalizeInput = nn.BatchNorm1d(1)
        
        self.conv1 = nn.Conv1d(1, 8, 65)
        self.ELU = nn.ReLU()

        self.conv2 = nn.Conv1d(8, 8, 33)

        self.maxpool1 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(8, 16, 33)

        self.conv4 = nn.Conv1d(16, 16, 17)

        self.maxpool2 = nn.MaxPool1d(3)

        self.conv5 = nn.Conv1d(16, 32, 17)
        
        self.conv6 = nn.Conv1d(32, 32, 17)

        self.maxpool3 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(1824, 64)
        self.dropout1 = nn.Dropout(p=0.5) #TODO: Decide p
        
        self.linear2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.5) #TODO: Decide p

        self.linear3 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Sigmoid()
        #self.unboundedSoftmax = nn.Linear(2, 2, bias=False)
        #self.unboundedSoftmax._parameters['weight'] = nn.Parameter(
        #        torch.Tensor([[0.0, -1.0], [-1.0, 0.0]]), requires_grad=False)

    def forward(self, inputs):
        #z = self.network(inputs)

        # Input layer
        z = self.normalizeInput(inputs)

        # 1. Layer - Conv1D + ELU
        z = self.conv1(z)
        z = self.ELU(z)
        
        # 2. Layer - Conv1D
        z = self.conv2(z)

        # 3. Layer - MaxPool1D + ELU
        z = self.maxpool1(z)
        z = self.ELU(z)

        # 4. Layer - Conv1D + ELU
        z = self.conv3(z)
        z = self.ELU(z)

        # 5. Layer - Conv+d
        z = self.conv4(z)

        # 6. Layer - MaxPool1D + ELU
        z = self.maxpool2(z)
        z = self.ELU(z)

        # 7. Layer - Conv1D + ELU
        z = self.conv5(z)
        z = self.ELU(z)

        # 8. Layer - Conv1D
        z = self.conv6(z)

        # 9. Layer - MaxPool1D + ELU
        z = self.maxpool3(z)
        z = self.ELU(z)

        # 10. Layer - Flatten
        z = self.flatten(z)

        # 11. Layer - Linear + Dropout + ELU
        z = self.linear1(z)
        z = self.dropout1(z)
        z = self.ELU(z)

        # 11. Layer - Linear + Dropout + ELU
        z = self.linear2(z)
        z = self.dropout2(z)
        z = self.ELU(z)

        # Output Layer - Linear + Softmax
        z = self.linear3(z)
        z = self.output(z)

        return z.unsqueeze(1)
