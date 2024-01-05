import torch
import torch.nn as nn


class Connect4Model(nn.Module):
    def __init__(
        self, board_shape=(6, 7, 3), num_players=3, num_moves=7, num_outcomes=7
    ):
        super(Connect4Model, self).__init__()
        print(f"board_shape {board_shape}")

        # Constants for dimensions
        in_channels = board_shape[0]
        num_conv_filters = 64
        num_fc_features = 128
        num_player_fc = 64
        num_move_fc = 64

        # Convolutional layers for processing the board
        self.board_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, num_conv_filters, kernel_size=3, padding=1
            ),  # Added padding=1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                num_conv_filters, 128, kernel_size=3, padding=3
            ),  # Added padding=1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers for processing the board features
        self.board_fc = nn.Sequential(
            nn.Linear(128 * 3 * 2, num_fc_features),  # Correct input size
            nn.ReLU(),
        )

        # Fully connected layers for processing current player, best move, and winner
        self.player_fc = nn.Sequential(nn.Linear(num_players, num_player_fc), nn.ReLU())
        self.move_fc = nn.Sequential(nn.Linear(num_moves, num_move_fc), nn.ReLU())

        # Final combined fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(256, num_fc_features),
            nn.ReLU(),
            nn.Linear(num_fc_features, num_outcomes),
        )

    def forward(self, board, current_player, best_move, winner):
        print(f"Input board shape: {board.shape}")
        print(f"current_player {current_player.shape}")
        print(f"best_move {best_move.shape}")
        print(f"winner {winner.shape}")

        # Process the board through convolutional and fully connected layers
        board = self.board_conv(board)
        print(f"Conv output shape: {board.shape}")
        board = board.view(board.size(0), 128 * 3 * 2)
        print(f"board after view {board.shape}")
        board = self.board_fc(board)
        print(f"FC output shape: {board.shape}")

        # Process current player, best move, and winner through fully connected layers
        player_features = self.player_fc(current_player)
        print(f"player_features {player_features.shape}")
        move_features = self.move_fc(best_move)
        print(f"move_features {move_features.shape}")

        # Concatenate all features
        combined_features = torch.cat((board, player_features, move_features), dim=1)
        print(f"Combined features shape: {combined_features.shape}")

        # Final prediction with softmax activation
        output = torch.softmax(self.fc(combined_features), dim=1)
        print(f"Output shape: {output.shape}")
        return output
