import os
import torch
from torch import optim
from Model.simple_model import SimpleNN
from Datasets.fash_mnist import get_data_loaders
from Training.train import train_model
from Analyze.utils import plot_training_progress
import datetime

def main():
    # Paths
    checkpoint_path = os.path.join("Model", "model_checkpoint.pth")
    log_file = os.path.join("Training", "training_log.txt")

    # Load dataset
    train_loader, _ = get_data_loaders(batch_size=64)

    # Initialize model and optimizer
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Loading model checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model and optimizer states loaded.")

    # Train the model and retrieve metrics
    train_losses, train_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=2,
        save_path=checkpoint_path,
        log_file=log_file
    )

    # Plot the training progress and save to the Analyze directory
    plot_training_progress(
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        save_dir="Analyze",
        file_name = f"training_progress_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )

if __name__ == "__main__":
    main()
