import argparse
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


from src.game.game import Game
from src.constants import P1, P2
from src.gui.bot_1 import BotPlayer
from src.gui.agent import AgentPlayer
from src.game.board import Board
from src.debug import dprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEvaluator:
    def __init__(self, models_dir, num_games=1):
        dprint("Initializing.")
        self.models_dir = models_dir
        self.num_games = num_games
        self.model_performance = {}
        self.invalid_moves_tracker = {}

    def evaluate_model(self, model_path):
        print(f"Evaluating model: {model_path}")
        self.wins, self.losses, self.draws, self.foreits = 0, 0, 0, 0
        self.player_2 = AgentPlayer(P2, model_path)
        self.reset_game()
        self.invalid_moves_tracker[model_path] = 0

        for game_number in tqdm(range(self.num_games), desc=f"Evaluating {model_path}"):
            dprint(f"Starting game {game_number + 1}...")
            self.run_single_game(model_path)
            winner = self.game.get_winner()
            dprint(f"Game ended. Winner: {winner}")
            self.update_counters(winner)
            self.reset_game()

        win_percentage, loss_percentage, draw_percentage = self.calculate_percentages()
        self.model_performance[model_path] = {
            "win": win_percentage,
            "loss": loss_percentage,
            "draw": draw_percentage,
            "forfeit": self.foreits,  # Assuming self.foreits is an integer count of forfeits
            "invalid_moves": self.invalid_moves_tracker[
                model_path
            ],  # This should be an int
        }

        self.plot_results()

    def reset_game(self):
        dprint("Resetting game...")
        self.player_1 = BotPlayer(P1)
        self.game = Game([self.player_1, self.player_2])
        self.player_1.set_game(self.game)
        self.player_2.set_game(self.game)

    def run_single_game(self, model_path):
        dprint("Running a single game...")
        invalid_moves = 0
        while not self.game.is_game_over():
            move_made = self.game.current_player.move(self.game.board)
            if move_made is None or not move_made in Board.get_valid_moves(
                self.game.board
            ):
                invalid_moves += 1
                self.invalid_moves_tracker[model_path] += invalid_moves
                dprint(
                    f"Invalid move by {self.game.current_player}. Count: {invalid_moves}"
                )
                if invalid_moves >= 4:
                    dprint("Four consecutive invalid moves made. Counting as loss.")
                    self.losses += 1
                    return
            else:
                invalid_moves = 0  # Reset counter if a valid move is made
                self.game.drop_piece(move_made, self.game.current_player.player_token)

    def update_counters(self, winner):
        dprint(f"Updating counters for winner: {winner}")
        if winner == P2:
            self.wins += 1
        elif winner == P1:
            self.losses += 1
        else:
            if self.game.is_draw():
                self.draws += 1
            else:
                self.foreits += 1

    def calculate_percentages(self):
        total_games = self.wins + self.losses + self.draws
        win_percentage = (self.wins / total_games) * 100 if total_games > 0 else 0
        loss_percentage = (self.losses / total_games) * 100 if total_games > 0 else 0
        draw_percentage = (self.draws / total_games) * 100 if total_games > 0 else 0
        return win_percentage, loss_percentage, draw_percentage

    def run(self):
        print("Starting model evaluation...")
        for filename in os.listdir(self.models_dir):
            if filename.endswith(".pth"):
                model_path = os.path.join(self.models_dir, filename)
                self.evaluate_model(model_path)
        print("Model evaluation plotted.")

    def plot_results(self):
        # Prepare data for plotting
        models = list(self.model_performance.keys())
        wins = [self.model_performance[model]["win"] for model in models]
        losses = [self.model_performance[model]["loss"] for model in models]
        draws = [self.model_performance[model]["draw"] for model in models]
        forfeits = [self.model_performance[model].get("forfeit", 0) for model in models]
        invalid_moves = [self.invalid_moves_tracker.get(model, 0) for model in models]

        # Plotting using line plot
        x = range(len(models))
        plt.figure(figsize=(10, 6))
        plt.plot(x, wins, marker="o", label="Wins")
        plt.plot(x, losses, marker="o", label="Losses")
        plt.plot(x, draws, marker="o", label="Draws")
        plt.plot(x, forfeits, marker="o", label="Forfeits")
        plt.plot(x, invalid_moves, marker="o", label="Invalid Moves")

        plt.xlabel("Models")
        plt.ylabel("Percentage")
        plt.title("Model Performance")
        plt.xticks(x, [os.path.basename(model) for model in models], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"models/evaluation.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument(
        "--models_dir",
        help="Model directory",
        required=False,
        default="./models",
        type=str,
    )
    parser.add_argument(
        "--num_games",
        help="How many games to play?",
        required=False,
        type=int,
        default=100,
    )
    args = parser.parse_args()

    evaluator = ModelEvaluator(
        models_dir=args.models_dir,
        num_games=args.num_games,
    )
    evaluator.run()
