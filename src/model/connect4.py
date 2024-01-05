import torch
import torch.nn as nn

from debug import dprint


class Connect4Model(nn.Module):
    def __init__(self, board_shape=(6, 7, 3), num_players=3, num_outcomes=7):
        super(Connect4Model, self).__init__()
        dprint(f"board_shape {board_shape}")

        in_channels = board_shape[0]
        num_conv_filters = 32
        num_fc_features = 64
        num_player_fc = 32

        # Convolutional layers for processing the board state
        self.board_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_conv_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_conv_filters, 128, kernel_size=3, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers for processing the board state after convolution
        self.board_fc = nn.Sequential(
            nn.Linear(128 * 3 * 2, num_fc_features),
            nn.ReLU(),
        )

        # Fully connected layers for processing the current player
        self.player_fc = nn.Sequential(nn.Linear(num_players, num_player_fc), nn.ReLU())

        # Output fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(num_fc_features + num_player_fc, num_fc_features),
            nn.ReLU(),
            nn.Linear(num_fc_features, num_outcomes),
        )

    def forward(self, board, current_player):
        dprint(f"Input board shape: {board.shape}")
        dprint(f"current_player {current_player.shape}")

        # Pass the board through convolutional layers
        board = self.board_conv(board)
        dprint(f"Conv output shape: {board.shape}")
        board = board.view(board.size(0), -1)
        dprint(f"board after view {board.shape}")

        # Pass the board through fully connected layers
        board_features = self.board_fc(board)
        dprint(f"Board FC output shape: {board_features.shape}")

        # Pass the current player through fully connected layers
        player_features = self.player_fc(current_player)
        dprint(f"Player features shape: {player_features.shape}")

        # Combine the board features with the player features
        combined_features = torch.cat((board_features, player_features), dim=1)
        dprint(f"Combined features shape: {combined_features.shape}")

        # Get the final output from the last fully connected layers
        output = self.fc(combined_features)
        dprint(f"Output shape: {output.shape}")
        return output
