import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from itertools import product


from src.model.connect4 import Connect4Model
from src.model.dataset import Connect4Dataset


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
    plt.savefig(f"./graphs/{filename}")
    plt.close()


# Load Dataset and DataLoader
train_dataset = Connect4Dataset(directory="./game_states", split="train")
validation_dataset = Connect4Dataset(directory="./game_states", split="validation")


num_conv_filters_sizes = [4, 8, 16, 32, 64]
num_fc_features_sizes = [4, 8, 16, 32, 64]
num_player_fc_sizes = [4, 8, 16, 32, 64]
num_game_state_fc_sizes = [4, 8, 16, 32, 64]

batch_sizes = [16, 32, 64, 128]
learning_rates = [0.001, 0.005, 0.0001]
num_epochs = 100

parameter_combinations = product(
    batch_sizes,
    learning_rates,
    num_conv_filters_sizes,
    num_fc_features_sizes,
    num_player_fc_sizes,
    num_game_state_fc_sizes,
)

for (
    batch_size,
    learning_rate,
    num_conv_filters_size,
    num_fc_features_size,
    num_player_fc_size,
    num_game_state_fc_size,
) in parameter_combinations:
    model_filename = f"connect4_model_{batch_size}_{learning_rate}_{num_conv_filters_size}_{num_fc_features_size}_{num_player_fc_size}.pth"

    if os.path.exists(f"./models/{model_filename}"):
        print(f"Skipping training for parameters that match model: {model_filename}")
        continue

    print(
        f"Running for bs: {batch_size}, lr: {learning_rate}, conv_filters_size: {num_conv_filters_size}, fc_features_size: {num_fc_features_size}, player_fc_size: {num_player_fc_size}, game_state_fc_size: {num_game_state_fc_size}"
    )

    # Define your model, criterion, and optimizer
    model = Connect4Model(
        board_shape=(6, 7, 3),
        num_players=3,
        num_outcomes=7,
        num_game_states=4,
        num_conv_filters=num_conv_filters_size,
        num_fc_features=num_fc_features_size,
        num_player_fc=num_player_fc_size,
        num_game_state_fc=num_game_state_fc_size,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    patience = 10
    best_validation_loss = float("inf")
    no_improvement_count = 0
    best_validation_accuracy = 0.0

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

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            board_states, current_players, best_moves, game_states = batch

            # Forward pass
            outputs = model(board_states, current_players, game_states)

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
                board_states, current_players, best_moves, game_states = batch

                # Forward pass for validation
                outputs = model(board_states, current_players, game_states)

                # Calculate the validation loss
                loss = criterion(
                    outputs,
                    torch.max(best_moves.view(best_moves.size(0), -1), 1)[1],
                )
                validation_loss += loss.item()

                # Get the predicted moves
                _, predicted_moves = torch.max(outputs, 1)

                # Get the true moves
                _, true_moves = torch.max(best_moves.view(best_moves.size(0), -1), 1)

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
        if accuracy > best_validation_accuracy:
            best_validation_accuracy = accuracy
            no_improvement_count = 0
            # Save model with the best accuracy
            torch.save(model.state_dict(), f"./models/{model_filename}")
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(
                f"Early stopping after {patience} epochs without improvement in accuracy."
            )
            break  # End training loop

        print(
            f"Epoch [{epoch+1:03d}/{num_epochs:03d}], Batch Size: {batch_size:03d}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_validation_loss:.4f}, Accuracy: {accuracy:.4f}, Learning Rate: {learning_rates[epoch]:.6f}, num_conv_filters_size: {num_conv_filters_size:04d}, num_fc_features_size: {num_fc_features_size:04d}, num_player_fc_size: {num_player_fc_size:04d}"
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
            f"learning_curves_{batch_size}_{learning_rate}_{num_conv_filters_size}_{num_fc_features_size}_{num_player_fc_size}.png",
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
                game_states,
            ) = batch

            # Predict the best move without providing the 'best_move'
            outputs = model(board_states, current_players, game_states)

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
