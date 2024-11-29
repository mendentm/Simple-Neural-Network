import torch
def train_model(
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        epochs=10, 
        save_path="model_checkpoint.pth", 
        log_file="training_log.txt"
        ):
    # Trains the model, saves the state after training, and logs results to a text file.
    train_losses = []
    train_accuracies = []

    # Open log file in append mode
    with open(log_file, "a") as log:
        for epoch in range(epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            model.train()  # Set model to training mode

            for images, labels in train_loader:
                optimizer.zero_grad()  # Reset gradients

                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss

                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                # Track loss and accuracy
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

            avg_loss = running_loss / len(train_loader)
            accuracy = correct_predictions / total_predictions
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)

            # Log results
            log_message = f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}\n"
            print(log_message.strip())
            log.write(log_message)
            

        # Save model and optimizer state after training
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accuracies': train_accuracies
        }, save_path)

        print(f"Model saved to {save_path}.\nTraining log saved to {log_file}.")
    return train_losses, train_accuracies
