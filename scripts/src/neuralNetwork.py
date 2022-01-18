from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Split network into features() and classifier() using
        # sequential module. This currently is a bit ugly.
        #self.network = get_network(path_to_weights)
        
        # Input Layer
        self.normalizeInput = nn.BatchNorm1d(1)
        
        # 1. Layer
        self.conv1 = nn.Conv1d(1, 8, 65)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.LeakyReLU1 = nn.LeakyReLU()
        
        # 2. Layer
        self.conv2 = nn.Conv1d(8, 8, 33)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.maxpool1 = nn.MaxPool1d(4)
        self.LeakyReLU2 = nn.LeakyReLU()
        
        # 3. Layer
        self.conv3 = nn.Conv1d(8, 16, 33)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.LeakyReLU3 = nn.LeakyReLU()

        # 4. Layer
        self.conv4 = nn.Conv1d(16, 16, 17)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.maxpool2 = nn.MaxPool1d(3)
        self.LeakyReLU4 = nn.LeakyReLU()
        
        # 5. Layer
        self.conv5 = nn.Conv1d(16, 32, 17)
        nn.init.xavier_uniform_(self.conv5.weight)
        self.LeakyReLU5 = nn.LeakyReLU()

        # 6. Layer
        self.conv6 = nn.Conv1d(32, 32, 17)
        nn.init.xavier_uniform_(self.conv6.weight)
        self.maxpool3 = nn.MaxPool1d(2)
        self.LeakyReLU6 = nn.LeakyReLU()
        
        # 7. Layer
        self.flatten = nn.Flatten()
        
        # 8. Layer
        self.linear1 = nn.Linear(1824, 64)
        self.dropout1 = nn.Dropout(p=0.5) #TODO: Decide p
        self.LeakyReLU7 = nn.LeakyReLU()
        
        # 9. Layer
        self.linear2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.5) #TODO: Decide p
        self.LeakyReLU8 = nn.LeakyReLU()

        self.linear3 = nn.Linear(64, 2)

        self.output= nn.Softmax(dim=1)

    def forward(self, inputs):
        #z = self.network(inputs)

        # Input layer
        z = self.normalizeInput(inputs)

        # 1. Layer - Conv1D + LeakyReLU
        z = self.conv1(z)
        z = self.LeakyReLU1(z)
        
        # 2. Layer - Conv1Da + MaxPool1D + LeakyReLU
        z = self.conv2(z)
        z = self.maxpool1(z)
        z = self.LeakyReLU2(z)

        # 3. Layer - Conv1D + LeakyReLU
        z = self.conv3(z)
        z = self.LeakyReLU3(z)

        # 4. Layer - Conv1d + MaxPool1D + LeakyReLU
        z = self.conv4(z)
        z = self.maxpool2(z)
        z = self.LeakyReLU4(z)

        # 5. Layer - Conv1D + LeakyReLU
        z = self.conv5(z)
        z = self.LeakyReLU5(z)

        # 6. Layer - Conv1D + MaxPool1D + LeakyReLU
        z = self.conv6(z)
        z = self.maxpool3(z)
        z = self.LeakyReLU6(z)

        # 7. Layer - Flatten
        z = self.flatten(z)

        # 8. Layer - Linear + Dropout + LeakyReLU
        z = self.linear1(z)
        z = self.dropout1(z)
        z = self.LeakyReLU7(z)

        # 9. Layer - Linear + Dropout + LeakyReLU
        z = self.linear2(z)
        z = self.dropout2(z)
        z = self.LeakyReLU8(z)

        # Output Layer - Linear + Softmax
        z = self.linear3(z)
        z = self.output(z)

        return z.unsqueeze(1)
