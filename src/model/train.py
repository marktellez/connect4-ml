import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model.connect4 import Connect4Model
from model.dataset import Connect4Dataset


# Define a function for plotting the learning curves side by side
def plot_learning_curves(
    train_losses, validation_losses, accuracies, learning_rates, filename
):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 6)
    )  # Create two side-by-side subplots

    # Plot training and validation losses on the first subplot
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    ax1.plot(
        range(1, len(validation_losses) + 1), validation_losses, label="Validation Loss"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs. Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy and learning rates on the second subplot
    ax2.plot(
        range(1, len(accuracies) + 1), accuracies, label="Accuracy", linestyle="--"
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    ax2_lr = ax2.twinx()  # Create a second y-axis for learning rates
    ax2_lr.plot(
        range(1, len(learning_rates) + 1),
        learning_rates,
        label="Learning Rate",
        color="red",
        linestyle=":",
    )
    ax2_lr.set_ylabel("Learning Rate", color="red")
    ax2_lr.legend(loc="upper right")

    plt.tight_layout()  # Ensure subplots don't overlap
    plt.savefig(filename)
    plt.close()


batch_sizes = [32, 64, 128]
learning_rates = [0.001, 0.001, 0.0001]

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        # Define your model, criterion, and optimizer
        model = Connect4Model(board_shape=(6, 7, 3), num_players=3, num_outcomes=7)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        model_path = "connect4_model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded pre-trained model from {model_path}")

        patience = 10
        best_validation_loss = float("inf")
        no_improvement_count = 0

        # Load Dataset and DataLoader
        train_dataset = Connect4Dataset(directory="./game_states", split="train")
        validation_dataset = Connect4Dataset(
            directory="./game_states", split="validation"
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

        print(f"Number of training examples: {len(train_dataset)}")
        print(f"Number of validation examples: {len(validation_dataset)}")

        # Lists to store training and validation losses
        train_losses = []
        validation_losses = []
        accuracies = []
        learning_rates = []

        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                board_states, current_players, best_moves, winners = batch

                # Forward pass
                outputs = model(board_states, current_players)

                # Calculate the loss
                loss = criterion(
                    outputs, torch.max(best_moves.view(best_moves.size(0), -1), 1)[1]
                )

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            # Validation step
            model.eval()
            with torch.no_grad():
                validation_loss = 0
                correct_predictions = 0
                total_predictions = 0

                for batch in validation_loader:
                    board_states, current_players, best_moves, winners = batch

                    # Forward pass for validation
                    outputs = model(board_states, current_players)

                    # Calculate the validation loss
                    loss = criterion(
                        outputs,
                        torch.max(best_moves.view(best_moves.size(0), -1), 1)[1],
                    )
                    validation_loss += loss.item()

                    # Get the predicted moves
                    _, predicted_moves = torch.max(outputs, 1)

                    # Get the true moves
                    _, true_moves = torch.max(
                        best_moves.view(best_moves.size(0), -1), 1
                    )

                    # Update correct predictions
                    correct_predictions += (predicted_moves == true_moves).sum().item()
                    total_predictions += true_moves.size(0)

            lr_scheduler.step()
            # Calculate and print the average loss for this epoch
            average_train_loss = total_loss / len(train_loader)
            average_validation_loss = validation_loss / len(validation_loader)

            accuracy = correct_predictions / total_predictions
            accuracies.append(accuracy)

            learning_rates.append(optimizer.param_groups[0]["lr"])

            # Check for early stopping
            if average_validation_loss < best_validation_loss:
                best_validation_loss = average_validation_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break  # End training loop

            print(
                f"Epoch [{epoch+1:03d}/{num_epochs:03d}], Batch Size: {batch_size:03d}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_validation_loss:.4f}, Accuracy: {accuracy:.4f}, Learning Rate: {learning_rates[epoch]:.4f}"
            )

            # Append losses to the lists
            train_losses.append(average_train_loss)
            validation_losses.append(average_validation_loss)

            # Call the function to plot learning curves and save
            plot_learning_curves(
                train_losses,
                validation_losses,
                accuracies,
                learning_rates,
                f"learning_curves_{batch_size}_{learning_rate}.png",
            )
            torch.save(
                model.state_dict(), f"connect4_model_{batch_size}_{learning_rate}.pth"
            )


def final_evaluation(model, validation_loader):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in validation_loader:
            (
                board_states,
                current_players,
                best_moves,
                _,
            ) = batch

            # Predict the best move without providing the 'best_move'
            outputs = model(board_states, current_players)

            # Get the predicted moves
            _, predicted_moves = torch.max(outputs, 1)

            # Get the true moves
            _, true_moves = torch.max(best_moves.view(best_moves.size(0), -1), 1)

            # Update correct predictions
            correct_predictions += (predicted_moves == true_moves).sum().item()
            total_predictions += true_moves.size(0)

    # Calculate final accuracy
    final_accuracy = correct_predictions / total_predictions
    return final_accuracy


final_accuracy = final_evaluation(model, validation_loader)
print(f"Final Evaluation Accuracy (without best_move): {final_accuracy:.4f}")
