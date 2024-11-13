import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer to hidden layer
        self.fc2 = nn.Linear(128, 10)       # Hidden layer to output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)            # Flatten the input
        x = F.relu(self.fc1(x))            # Apply ReLU activation
        x = self.fc2(x)                    # Output layer
        return x
