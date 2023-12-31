import sys
from tqdm import tqdm
import copy
import json
import argparse
import torch
import pygame
import os
import time

from src.game.game import Game
from src.constants import P1, P2, E0
from src.model.ohe import move_to_ohe, winner_to_ohe, valid_moves_to_ohe


from src.gui.board import GUIBoard
from src.gui.bot_1 import MCTSBotPlayer
from src.gui.bot_2 import ProcBotPlayer
from src.gui.human import HumanPlayer
from src.gui.agent import AgentPlayer
from src.model.connect4 import Connect4Model
from src.game.board import Board

from src.debug import dprint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_path(model_path):
    if not os.path.exists(model_path):
        print(f"Model checkpoint {model_path} not found.")
        return None

    try:
        model = Connect4Model()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


class GUI:
    def __init__(
        self,
        model_path=None,
        bot_vs_bot=False,
        bot_vs_agent=False,
        human_vs_agent=False,
        agent_vs_agent=False,
        no_gui=False,
        num_games=1,
    ):
        print(f"bot_vs_bot {bot_vs_bot}")
        print(f"bot_vs_agent {bot_vs_agent}")
        pygame.init()

        self.WIDTH, self.HEIGHT = 700, 600
        self.SQUARE_SIZE = self.WIDTH // 7

        self.board_ui = GUIBoard(self.WIDTH, self.HEIGHT, self.SQUARE_SIZE)

        self.clock = pygame.time.Clock()

        self.num_games = num_games
        self.game_states = []
        self.winning_combo = []

        self.total_games = 0
        self.predicted_winner = None
        self.bot_vs_agent = bot_vs_agent
        self.bot_vs_bot = bot_vs_bot
        self.human_vs_agent = human_vs_agent
        self.agent_vs_agent = agent_vs_agent
        self.model_path = model_path
        self.no_gui = no_gui

        if not self.no_gui:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Connect Four")

        self.progress_bar = tqdm(
            total=self.num_games, disable=not self.no_gui, desc="Simulating games"
        )

        self.games_written = 0

    def update_display(self):
        if not self.no_gui:
            self.board_ui.draw(self.game.board, self.screen)
            pygame.display.update()

    def reset(self):
        if args.bot_vs_bot:
            dprint(f"Bot vs Bot")
            self.player_1 = MCTSBotPlayer(P1)
            self.player_2 = ProcBotPlayer(P2)

        elif self.model_path is not None and self.bot_vs_agent:
            dprint(f"Bot vs Agent")
            self.player_1 = MCTSBotPlayer(P1)
            self.player_2 = AgentPlayer(P2, self.model_path)

        elif self.model_path is not None and self.human_vs_agent:
            dprint(f"Human vs Agent")
            self.player_1 = HumanPlayer(P1, self.SQUARE_SIZE)
            self.player_2 = AgentPlayer(P2, self.model_path)

        elif self.model_path is not None and self.agent_vs_agent:
            dprint(f"Agent vs Agent")
            self.player_1 = AgentPlayer(P1, self.model_path)
            self.player_2 = AgentPlayer(P2, self.model_path)

        else:
            print(f"Human vs Bot")
            self.player_1 = HumanPlayer(P1, self.SQUARE_SIZE)
            self.player_2 = MCTSBotPlayer(P2)

        self.game = Game([self.player_1, self.player_2])

        if (
            self.player_1.__class__ == AgentPlayer
            or self.player_1.__class__ == MCTSBotPlayer
            or self.player_1.__class__ == ProcBotPlayer
        ):
            self.player_1.set_game(self.game)

        if (
            self.player_2.__class__ == AgentPlayer
            or self.player_2.__class__ == MCTSBotPlayer
            or self.player_2.__class__ == ProcBotPlayer
        ):
            self.player_2.set_game(self.game)

        self.games_written = 0

    def run(self):
        run = True
        games_simulated = 0
        self.reset()

        try:
            while run:
                self.update_display()
                if games_simulated >= self.num_games:
                    break

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False

                if self.game.is_game_over():
                    winner = self.game.get_winner()
                    dprint(f"Winner {winner}")

                    self.save_game_state()

                    if self.winning_combo:
                        self.highlight_winning_combo()

                    games_simulated += 1

                    self.game = Game([self.player_1, self.player_2])

                    self.update_display()
                    self.clock.tick(60)
                    self.reset()

                else:
                    if isinstance(self.game.current_player, HumanPlayer):
                        move_made = None

                        # Wait for human player's input
                        while move_made is None:
                            self.board_ui.draw(self.game.board, self.screen)

                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    run = False
                                elif event.type == pygame.MOUSEBUTTONDOWN:
                                    # Handle mouse click to make a move
                                    move_made = self.game.current_player.move(event.pos)

                    else:
                        dprint(f"self.game.board {self.game.board}")
                        move_made = self.game.current_player.move(self.game.board)

                    if move_made != None:
                        if (self.bot_vs_bot or self.bot_vs_agent) and args.sleep:
                            time.sleep(1)

                        board_state_copy = copy.deepcopy(self.game.board)
                        current_player = self.game.get_current_player().player_token
                        valid_moves = valid_moves_to_ohe(
                            Board.get_valid_moves(board_state_copy)
                        )

                        self.game.drop_piece(
                            move_made, self.game.current_player.player_token
                        )

                        self.game_states.append(
                            {
                                "board_state": board_state_copy,
                                "current_player": current_player,
                                "best_move": move_to_ohe(move_made),
                                "game_state": winner_to_ohe(
                                    self.game.get_winner(), self.game.is_draw()
                                ),
                                "valid_moves": valid_moves,
                            }
                        )

            sys.stdout.flush()
            self.update_display()

        except KeyboardInterrupt:
            print("Game interrupted.")
        finally:
            pygame.quit()
            self.progress_bar.close()
            print("Game has been closed.")

    def save_game_state(self):
        save_dir = "./game_states"
        os.makedirs(save_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(save_dir) if f.endswith(".json")]
        if existing_files:
            highest_num = max([int(f.split(".")[0]) for f in existing_files])
            next_game_number = highest_num + 1
        else:
            next_game_number = 1

        # Define the directory where game states will be saved
        filename = os.path.join(save_dir, f"{next_game_number}.json")

        with open(filename, "w") as file:
            json.dump(self.game_states, file, indent=2)
        dprint(f"Saved game {next_game_number} to {filename}")

        self.games_written += 1
        self.progress_bar.update(self.games_written)
        self.game_states = []

    def get_empty_board(self):
        """
        Create and return an empty game board.
        """
        return [[E0 for _ in range(7)] for _ in range(6)]

    def find_horizontal_win(self):
        for row in range(6):
            for col in range(7 - 3):
                if self.game.board[row][col] != E0 and all(
                    self.game.board[row][col] == self.game.board[row][col + i]
                    for i in range(1, 4)
                ):
                    return [(row, col + i) for i in range(4)]
        return None

    def find_vertical_win(self):
        for col in range(7):
            for row in range(6 - 3):
                if self.game.board[row][col] != E0 and all(
                    self.game.board[row][col] == self.game.board[row + i][col]
                    for i in range(1, 4)
                ):
                    return [(row + i, col) for i in range(4)]
        return None

    def find_diagonal_right_win(self):
        for row in range(6 - 3):
            for col in range(7 - 3):
                if self.game.board[row][col] != E0 and all(
                    self.game.board[row + i][col + i] == self.game.board[row][col]
                    for i in range(1, 4)
                ):
                    return [(row + i, col + i) for i in range(4)]
        return None

    def find_diagonal_left_win(self):
        for row in range(3, 6):
            for col in range(7 - 3):
                if self.game.board[row][col] != E0 and all(
                    self.game.board[row - i][col + i] == self.game.board[row][col]
                    for i in range(1, 4)
                ):
                    return [(row - i, col + i) for i in range(4)]
        return None

    def find_winning_combo(self):
        return (
            self.find_horizontal_win()
            or self.find_vertical_win()
            or self.find_diagonal_right_win()
            or self.find_diagonal_left_win()
        )

    def get_highlight_positions(self, winning_combo):
        highlight_positions = []
        for r, c in winning_combo:
            center_x = int(c * self.SQUARE_SIZE + self.SQUARE_SIZE // 2)
            center_y = int(r * self.SQUARE_SIZE + self.SQUARE_SIZE // 2)
            highlight_positions.append((center_x, center_y))
        return highlight_positions

    def draw_highlights(self, highlight_positions):
        for position in highlight_positions:
            pygame.draw.circle(
                self.screen,
                (0, 255, 0),
                position,
                (self.SQUARE_SIZE // 2 - 5) // 2,
            )

    def highlight_winning_combo(self):
        if not self.winning_combo:
            return []

        highlight_positions = self.get_highlight_positions(self.winning_combo)
        self.draw_highlights(highlight_positions)
        return highlight_positions

    def reset_game(self):
        self.game.reset_board()
        self.player = P1
        self.winning_combo = []
        self.agent.games_played += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect Four GUI with AI")
    parser.add_argument(
        "--model_path", help="Path to the trained model checkpoint", required=False
    )
    parser.add_argument(
        "--num_games", help="How many games to play?", required=False, type=int
    )
    parser.add_argument(
        "--bot_vs_bot",
        help="Play bot vs bot",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--human_vs_bot",
        help="Play human vs bot",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--bot_vs_agent",
        help="Play bot vs agent",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--human_vs_agent",
        help="Play human vs agent",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--agent_vs_agent",
        help="Play agent vs agent",
        action="store_true",
        required=False,
        default=False,
    )

    parser.add_argument(
        "--sleep",
        help="Pause bot play for verification",
        action="store_true",
        required=False,
        default=False,
    )

    parser.add_argument(
        "--no_gui",
        help="Hide gui",
        action="store_true",
        required=False,
        default=False,
    )

    args = parser.parse_args()

    gui = GUI(
        model_path=args.model_path,
        bot_vs_bot=args.bot_vs_bot,
        bot_vs_agent=args.bot_vs_agent,
        human_vs_agent=args.human_vs_agent,
        agent_vs_agent=args.agent_vs_agent,
        no_gui=args.no_gui,
        num_games=1 if args.num_games == None else args.num_games,
    )
    gui.run()
