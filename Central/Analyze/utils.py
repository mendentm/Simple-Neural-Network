import matplotlib.pyplot as plt

def plot_training_progress(train_losses, train_accuracies):
    """Plots training loss and accuracy."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, '-o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")


    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, '-o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")


    plt.tight_layout()
    plt.show()


