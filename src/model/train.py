import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.connect4 import Connect4Model
from model.dataset import Connect4Dataset

# Define your model, criterion, and optimizer
model = Connect4Model(board_shape=(6, 7, 3), num_players=3, num_moves=7, num_outcomes=7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load Dataset and DataLoader
train_dataset = Connect4Dataset(directory="./game_states", split="train")
validation_dataset = Connect4Dataset(directory="./game_states", split="validation")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        board_states, current_players, best_moves, winners = batch
        print(
            f"board_states {board_states.shape}, current_players {current_players.shape}, best_moves {best_moves.shape}, winners {winners.shape}"
        )

        # Forward pass
        outputs = model(board_states, current_players, best_moves, winners)

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
        for batch in validation_loader:
            board_states, current_players, best_moves, winners = batch

            # Forward pass for validation
            outputs = model(board_states, current_players, best_moves, winners)

            # Calculate the validation loss
            loss = criterion(
                outputs, torch.max(best_moves.view(best_moves.size(0), -1), 1)[1]
            )
            validation_loss += loss.item()

    # Calculate and print the average loss for this epoch
    average_train_loss = total_loss / len(train_loader)
    average_validation_loss = validation_loss / len(validation_loader)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_train_loss:.4f}, Validation Loss: {average_validation_loss:.4f}"
    )
