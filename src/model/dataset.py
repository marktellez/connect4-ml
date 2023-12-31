import os
import torch
import json
import random
from torch.utils.data import Dataset


class Connect4Dataset(Dataset):
    def __init__(self, directory, split="train", split_ratio=0.8):
        self.data = []
        data_files = os.listdir(directory)
        random.seed(42)  # Fix the seed for reproducibility
        random.shuffle(data_files)  # Shuffle file names, not the data itself

        for filename in data_files:
            if filename.endswith(".json"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r") as file:
                    game_data = json.load(file)
                    self.data.extend(game_data)

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
        game_state = torch.tensor(self.data[idx]["game_state"], dtype=torch.float)
        valid_moves = torch.tensor(self.data[idx]["valid_moves"], dtype=torch.float)

        return board_state, current_player, best_move, game_state, valid_moves
