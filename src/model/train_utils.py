import sys
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.debug import dprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def final_evaluation(model, validation_loader):
    dprint(f"final_evaluation")
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in validation_loader:
            board_states, current_players, best_moves, game_states, valid_moves = [
                b.to(device) for b in batch
            ]
            outputs = model(board_states, current_players, game_states, valid_moves)
            _, predicted_moves = torch.max(outputs, 1)
            _, true_moves = torch.max(best_moves.view(best_moves.size(0), -1), 1)
            correct_predictions += (predicted_moves == true_moves).sum().item()
            total_predictions += true_moves.size(0)

    final_accuracy = correct_predictions / total_predictions
    return final_accuracy


def plot_learning_curves(
    train_losses, validation_losses, accuracies, learning_rates, filename
):
    dprint(f"plot_learning_curves")
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    ax1.plot(train_losses, label="Training Loss")
    ax1.plot(validation_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs. Validation Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(accuracies, label="Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True)
    ax3.plot(learning_rates, label="Learning Rate")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate")
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    graphs_dir = "./graphs"
    os.makedirs(graphs_dir, exist_ok=True)
    plt.savefig(f"{graphs_dir}/{filename}.png")
    plt.close()


def train_model_with_hyperparams(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    lr_scheduler,
    num_epochs,
    model_filename,
):
    dprint(f"train_model_with_hyperparams")
    train_loader_len = len(train_loader)
    validation_loader_len = len(validation_loader)
    train_losses = []
    validation_losses = []
    accuracies = []
    lr_rates = []
    best_validation_loss = float("inf")

    for epoch in range(num_epochs):
        dprint(f"Training epoch {epoch}")
        model.train()
        total_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"E{epoch+1}/{num_epochs} [Train]",
            total=len(train_loader),
            leave=False,
        )

        try:
            for batch in train_loader:
                optimizer.zero_grad()
                board_states, current_players, best_moves, game_states, valid_moves = [
                    b.to(device) for b in batch
                ]
                outputs = model(board_states, current_players, game_states, valid_moves)
                loss = criterion(
                    outputs,
                    torch.max(best_moves.view(best_moves.size(0), -1), 1)[1],
                    game_states,
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                train_bar.set_postfix(
                    loss=f"{loss.item():.2f}"  # Limit to 2 decimal places
                )
                train_bar.update()
                sys.stdout.flush()
            train_bar.close()

        except Exception as e:
            print(f"An error occurred during training: {e}")

        model.eval()
        validation_loss = 0
        correct_predictions = 0
        total_predictions = 0
        validation_bar = tqdm(
            validation_loader,
            desc=f"E{epoch+1}/{num_epochs} [Val]",
            total=len(validation_loader),
            leave=False,
        )

        with torch.no_grad():
            for batch in validation_loader:
                board_states, current_players, best_moves, game_states, valid_moves = [
                    b.to(device) for b in batch
                ]
                outputs = model(board_states, current_players, game_states, valid_moves)
                loss = criterion(
                    outputs,
                    torch.max(best_moves.view(best_moves.size(0), -1), 1)[1],
                    game_states,
                )
                validation_loss += loss.item()
                _, predicted_moves = torch.max(outputs, 1)
                _, true_moves = torch.max(best_moves.view(best_moves.size(0), -1), 1)
                correct_predictions += (predicted_moves == true_moves).sum().item()
                total_predictions += true_moves.size(0)
                validation_bar.set_postfix(
                    acc=f"{100.0 * (correct_predictions / total_predictions):.1f}%",  # Limit to 1 decimal place
                    v_loss=f"{loss.item():.2f}",  # Limit to 2 decimal places
                )
                validation_bar.update()
                sys.stdout.flush()
            validation_bar.close()

        lr_scheduler.step()
        average_train_loss = total_loss / train_loader_len
        average_validation_loss = validation_loss / validation_loader_len
        accuracy = correct_predictions / total_predictions
        lr_rates.append(optimizer.param_groups[0]["lr"])

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            torch.save(model.state_dict(), f"./models/{model_filename}")

        train_losses.append(average_train_loss)
        validation_losses.append(average_validation_loss)
        accuracies.append(accuracy)

        plot_learning_curves(
            train_losses, validation_losses, accuracies, lr_rates, model_filename
        )

    final_accuracy = final_evaluation(model, validation_loader)
    print(f"Final Evaluation Accuracy: {final_accuracy:.4f}")
