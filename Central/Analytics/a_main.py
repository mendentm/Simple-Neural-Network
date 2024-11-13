import matplotlib.pyplot as plt
from Training.train_eval import train_losses, train_accuracies

def plot_training_progress():
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Plotting loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, '-o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, '-o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.show()

# Call the plotting function
plot_training_progress()
