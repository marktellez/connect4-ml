import os
import torch
import json
import random
from torch.utils.data import Dataset


class Connect4Dataset(Dataset):
    def __init__(self, directory, split="train", split_ratio=0.8):
        self.data = []
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                with open(os.path.join(directory, filename), "r") as file:
                    game_data = json.load(file)
                    self.data.extend(game_data)

        random.shuffle(self.data)
        split_index = int(len(self.data) * split_ratio)
        if split == "train":
            self.data = self.data[:split_index]
        else:
            self.data = self.data[split_index:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_state = torch.tensor(self.data[idx]["board_state"], dtype=torch.float)
        current_player = torch.tensor(
            self.data[idx]["current_player"], dtype=torch.float
        )
        best_move = torch.tensor(self.data[idx]["best_move"], dtype=torch.float)
        winner = torch.tensor(self.data[idx]["winner"], dtype=torch.float)

        return board_state, current_player, best_move, winner
