import numpy as np
import torch
from torchvision import transforms

from src.model.connect4 import Connect4Model
from src.gui.player import Player
from src.constants import P1, P2

from src.model.ohe import winner_to_ohe


class AgentPlayer(Player):
    def __init__(self, player_token, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Connect4Model(
            board_shape=(6, 7, 3),
            num_players=3,
            num_outcomes=7,
            num_conv_filters=16,
            num_fc_features=16,
            num_player_fc=16,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.player_token = player_token
        self.opponent_token = P1 if player_token == P2 else P2
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.board_history = []

    def set_game(self, game):
        self.game = game

    def move(self, board):
        print(f"board {board}")
        board_state = torch.tensor(
            board, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        current_player = torch.tensor(
            self.player_token,
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)

        winner = torch.tensor(
            winner_to_ohe(self.game.get_winner()), dtype=torch.float, device=self.device
        ).unsqueeze(0)

        print(f"board_state {board_state.shape}")
        print(f"current_player {current_player.shape}")

        print(f"winner {winner.shape}")

        # Pass tensors to the model
        predictions = self.model(board_state, current_player)

        print(f"predictions {predictions}")

        predictions_cpu = predictions.detach().cpu().numpy()
        predicted_move = np.argmax(predictions_cpu[0])

        return predicted_move
