import re
import numpy as np
import torch
from torchvision import transforms

from src.model.connect4 import Connect4Model
from src.gui.player import Player
from src.constants import P1, P2
from src.game.board import Board

from src.model.ohe import winner_to_ohe, valid_moves_to_ohe

from src.debug import dprint


class AgentPlayer(Player):
    def __init__(self, player_token, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        (
            _,
            _,
            num_conv_filters,
            num_fc_features,
            num_player_fc,
            num_game_state_fc,
        ) = self.parse_model_filename(model_path)

        self.model = Connect4Model(
            board_shape=(6, 7, 3),
            num_players=3,
            num_outcomes=7,
            num_game_states=4,
            num_conv_filters=num_conv_filters,
            num_fc_features=num_fc_features,
            num_player_fc=num_player_fc,
            num_game_state_fc=num_game_state_fc,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.player_token = player_token
        self.opponent_token = P1 if player_token == P2 else P2
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.board_history = []

    def parse_model_filename(self, model_path):
        # Updated pattern to include exponential notation for the learning rate
        pattern = r"connect4_model_(\d+)_([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)_(\d+)_(\d+)_(\d+)_(\d+).*\.pth"
        match = re.search(pattern, model_path)
        if match:
            batch_size = int(match.group(1))
            learning_rate = float(match.group(2))
            num_conv_filters = int(match.group(3))
            num_fc_features = int(match.group(4))
            num_player_fc = int(match.group(5))
            num_game_state_fc = int(match.group(6))
            return (
                batch_size,
                learning_rate,
                num_conv_filters,
                num_fc_features,
                num_player_fc,
                num_game_state_fc,
            )
        else:
            raise ValueError("Model filename does not match expected pattern")

    def set_game(self, game):
        self.game = game

    def move(self, board):
        dprint(f"board {board}")
        board_state = torch.tensor(
            board, dtype=torch.float, device=self.device
        ).unsqueeze(0)

        current_player = torch.tensor(
            self.player_token,
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)

        game_state = torch.tensor(
            winner_to_ohe(self.game.get_winner()), dtype=torch.float, device=self.device
        ).unsqueeze(0)

        valid_moves = torch.tensor(
            valid_moves_to_ohe(Board.get_valid_moves(self.game.board)),
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)

        dprint(f"board_state {board_state.shape}")
        dprint(f"current_player {current_player.shape}")

        dprint(f"game_state {game_state.shape}")

        # Pass tensors to the model
        predictions = self.model(board_state, current_player, game_state, valid_moves)

        dprint(f"predictions {predictions}")

        predictions_cpu = predictions.detach().cpu().numpy()
        print(f"predictions_cpu {predictions_cpu}")
        predicted_move = np.argmax(predictions_cpu[0])

        return predicted_move
