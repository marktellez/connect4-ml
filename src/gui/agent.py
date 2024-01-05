import numpy as np
import torch
from torchvision import transforms

from model.connect4 import Connect4Model
from player import Player
from constants import P1, P2


class AgentPlayer(Player):
    def __init__(self, player_token, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Connect4Model().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.player_token = player_token
        self.opponent_token = P1 if player_token == P2 else P2
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.board_history = []

    def set_game(self, game):
        self.game = game

    def move(self, board):
        board_state = torch.tensor(
            board, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        current_player = torch.tensor(
            self.player_token,
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0)
        best_move = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device
        )

        print(f"board_state {board_state.shape}")
        print(f"current_player {current_player.shape}")
        print(f"best_move {best_move.shape}")

        # Pass tensors to the model
        move_logits, policy_output = self.model(
            board_state,
            current_player,
            best_move,
        )

        print(f"move_logits {move_logits}")
        print(f"policy_output {policy_output}")

        predictions_cpu = policy_output.detach().cpu().numpy()
        predicted_move = np.argmax(predictions_cpu[0])

        return predicted_move
