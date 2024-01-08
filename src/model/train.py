import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from itertools import product
import argparse


from src.model.connect4 import Connect4Model
from src.model.custom_loss import CustomLoss
from src.model.dataset import Connect4Dataset
from src.model.train_utils import train_model_with_hyperparams

from src.debug import set_seed, dprint

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument("--num_epochs", default=100, type=int)
args = parser.parse_args()

train_dataset = Connect4Dataset(
    directory="./training_data", split="train", split_ratio=0.7
)
validation_dataset = Connect4Dataset(
    directory="./training_data", split="validation", split_ratio=0.7
)

batch_sizes = [256]
learning_rates = [0.00001]
num_epochs = args.num_epochs
weight_decay = 1e-3


parameter_combinations = product(
    batch_sizes, learning_rates, [6 * 7 * 3 * 50], [6 * 7 * 3 * 50], [4 * 50], [4 * 50]
)

for (
    batch_size,
    learning_rate,
    num_conv_filters_size,
    num_fc_features_size,
    num_player_fc_size,
    num_game_state_fc_size,
) in parameter_combinations:
    model_filename = f"connect4_model_{batch_size}_{learning_rate}_{num_conv_filters_size}_{num_fc_features_size}_{num_player_fc_size}_{num_game_state_fc_size}.pth"

    # if os.path.exists(f"./models/{model_filename}"):
    #    continue

    model = Connect4Model(
        board_shape=(6, 7, 3),
        num_players=3,
        num_outcomes=7,
        num_game_states=4,
        num_conv_filters=num_conv_filters_size,
        num_fc_features=num_fc_features_size,
        num_player_fc=num_player_fc_size,
        num_game_state_fc=num_game_state_fc_size,
    ).to(device)

    criterion = CustomLoss(
        original_loss_weight=1.0,
        winning_reward=1.0,
        losing_reward=-1.0,
        draw_reward=0.0,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.95, 0.995),
        eps=1e-08,
        weight_decay=weight_decay,
    )
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.99)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    print(
        f"Setting up training session for {len(train_loader)*batch_size} training and {len(validation_loader)*batch_size} validation examples."
    )

    train_model_with_hyperparams(
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        lr_scheduler,
        num_epochs,
        model_filename,
    )
