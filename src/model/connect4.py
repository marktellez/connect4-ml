# Input board shape: torch.Size([64, 6, 7, 3])
# current_player torch.Size([64, 3])
# Conv output shape: torch.Size([64, 128, 3, 2])
# board after view torch.Size([64, 768])
# Board FC output shape: torch.Size([64, 512])
# Player features shape: torch.Size([64, 256])
# Game state features shape: torch.Size([64, 256])
# Combined features shape: torch.Size([64, 1024])
# Output shape: torch.Size([64, 7])


import torch
import torch.nn as nn

from src.debug import dprint, set_seed

set_seed()


class Connect4Model(nn.Module):
    def __init__(
        self,
        board_shape=(6, 7, 3),
        num_players=3,
        num_outcomes=7,
        num_game_states=4,
        num_valid_moves=7,
        num_conv_filters=32,
        num_fc_features=64,
        num_player_fc=32,
        num_game_state_fc=32,
        num_valid_moves_fc=16,
    ):
        super(Connect4Model, self).__init__()
        dprint(f"board_shape {board_shape}")

        in_channels = board_shape[0]

        self.dropout = nn.Dropout(0.5)

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

        # Fully connected layers for processing the game_state
        self.game_state_fc = nn.Sequential(
            nn.Linear(num_game_states, num_game_state_fc), nn.ReLU()
        )

        # Fully connected layers for processing the valid moves
        self.valid_moves_fc = nn.Sequential(
            nn.Linear(num_valid_moves, num_valid_moves_fc), nn.ReLU()
        )

        # Output fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(
                num_fc_features
                + num_player_fc
                + num_game_state_fc
                + num_valid_moves_fc,
                num_fc_features,
            ),
            nn.ReLU(),
            nn.Linear(num_fc_features, num_outcomes),
        )

    def forward(self, board, current_player, game_state, valid_moves):
        dprint(f"Input board shape: {board.shape}")
        dprint(f"current_player {current_player.shape}")

        # Pass the board through convolutional layers
        board = self.board_conv(board)
        board = self.dropout(board)
        dprint(f"Conv output shape: {board.shape}")
        board = board.view(board.size(0), -1)
        dprint(f"board after view {board.shape}")

        # Pass the board through fully connected layers
        board_features = self.board_fc(board)
        board_features = self.dropout(board_features)
        dprint(f"Board FC output shape: {board_features.shape}")

        # Pass the current player through fully connected layers
        player_features = self.player_fc(current_player)
        player_features = self.dropout(player_features)
        dprint(f"Player features shape: {player_features.shape}")

        # Pass the game state through fully connected layers
        game_state_features = self.game_state_fc(game_state)
        game_state_features = self.dropout(game_state_features)
        dprint(f"Game state features shape: {game_state_features.shape}")

        # Pass the valid moves through fully connected layers
        valid_moves_features = self.valid_moves_fc(valid_moves)
        valid_moves_features = self.dropout(valid_moves_features)
        dprint(f"Valid moves features shape: {valid_moves_features.shape}")

        # Combine the board features with the player features
        combined_features = torch.cat(
            (
                board_features,
                player_features,
                game_state_features,
                valid_moves_features,
            ),
            dim=1,
        )
        dprint(f"Combined features shape: {combined_features.shape}")

        # Get the final output from the last fully connected layers
        output = self.fc(combined_features)
        dprint(f"Output shape: {output.shape}")

        masked_output = output * valid_moves
        dprint(f"Masked output shape: {masked_output.shape}")

        return masked_output
