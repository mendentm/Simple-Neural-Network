import matplotlib.pyplot as plt
import os

def plot_training_progress(train_losses, train_accuracies, save_dir="Analyze", file_name="training_progress.png"):
    # Generate the next sequential file name
    existing_files = [f for f in os.listdir(save_dir) if f.startswith("training_progress_") and f.endswith(".png")]
    numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files if f.split("_")[2].split(".")[0].isdigit()]
    next_number = max(numbers, default=0) + 1
    file_name = f"training_progress_{next_number}.png"
    
    # Plots training loss and accuracy
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, '-o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, '-o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.grid(True)

    # Layout 
    plt.tight_layout()
    
    # Save plots
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close

