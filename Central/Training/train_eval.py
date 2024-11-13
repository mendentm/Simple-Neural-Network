import torch.nn as nn
import torch
from NN.net_main import NeuralNetwork
from Data.dataset_fash_mnist import train_loader

# Initialize model, loss function, and optimizer
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Lists to store training progress
train_losses = []
train_accuracies = []

# Training function
def train(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        model.train()
        
        for images, labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
        # Calculate average loss and accuracy for the epoch
        avg_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        # Store losses and accuracies for later plotting
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Train the model for 10 epochs (or adjust as needed)
train(model, train_loader, criterion, optimizer, epochs=10)
