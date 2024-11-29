import torch.nn as nn
import torch.nn.functional as F

# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)  # Flatten input
#         x = F.relu(self.fc1(x))  # Hidden layer with ReLU
#         x = self.fc2(x)          # Output layer
#         return x

import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 14 * 14, 128)  # Output of convolutional and pooling layers is 32 x 14 x 14
        self.fc2 = nn.Linear(128, 10)  # 10 output classes

    def forward(self, x):
        # Convolutional layer with ReLU activation
        x = F.relu(self.conv1(x))
        # Max-pooling
        x = self.pool(x)
        # Flatten layer
        x = x.view(x.size(0), -1)  # Flatten input
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x
