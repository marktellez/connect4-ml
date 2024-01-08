import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(
        self,
        original_loss_weight=1.0,
        winning_reward=1.0,
        losing_reward=-1.0,
        draw_reward=0.0,
    ):
        super(CustomLoss, self).__init__()
        self.original_loss_weight = original_loss_weight
        self.winning_reward = winning_reward
        self.losing_reward = losing_reward
        self.draw_reward = draw_reward

    def forward(self, predicted_moves, actual_moves, game_states):
        # Calculate the original loss (e.g., CrossEntropyLoss)
        original_loss = nn.CrossEntropyLoss()(predicted_moves, actual_moves)

        # Calculate the batch size
        batch_size = predicted_moves.size(0)

        # Reshape the predicted_moves and actual_moves to match batch_size
        predicted_moves = predicted_moves.view(batch_size, -1)
        actual_moves = actual_moves.view(batch_size, -1)

        # Calculate the reward based on game_states
        reward = torch.zeros_like(game_states, dtype=torch.float)

        # Here, you can implement custom logic based on the game_states
        # Assign rewards based on the desired outcomes

        # Example: Assign rewards based on game_states (modify this logic as needed)
        reward[game_states == 1] = self.winning_reward
        reward[game_states == -1] = self.losing_reward
        reward[game_states == 0] = self.draw_reward

        # Calculate the mean reward
        mean_reward = reward.mean()

        # Combine the original loss and the mean reward, weighted accordingly
        loss = (self.original_loss_weight * original_loss) - mean_reward

        return loss
