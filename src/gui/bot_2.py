import random
from src.gui.player import Player
from src.debug import dprint
from src.constants import P1, P2, E0
from src.game.board import Board


class ProcBotPlayer(Player):
    def __init__(self, player_token):
        super().__init__(player_token)

    def set_game(self, game):
        dprint(f"Setting game")
        self.game = game

    def move(self, board):
        # Check for each condition sequentially
        for check in [
            self.find_win_move,
            self.find_block_win_move,
            self.find_join_3,
            self.find_block_3,
            self.find_join_2,
            self.find_random_move,
        ]:
            move = check(board)
            if move is not None:
                return move

    def find_win_move(self, board):
        for col in range(7):
            if Board.is_column_not_full(col, board):
                simulated_board = Board.simulate_move(col, self.player_token, board)
                if Board.is_potential_win_board(simulated_board, self.player_token):
                    return col
        return None

    def find_block_win_move(self, board):
        opponent = P1 if self.player_token == P2 else P2
        for col in range(7):
            if Board.is_column_not_full(col, board):
                simulated_board = Board.simulate_move(col, self.player_token, board)
                if Board.is_potential_win_board(simulated_board, opponent):
                    return col
        return None

    def find_join_3(self, board):
        for col in range(7):
            if Board.is_column_not_full(col, board):
                simulated_board = Board.simulate_move(col, self.player_token, board)
                if (
                    Board.evaluate_join_potential(
                        simulated_board, self.player_token, [0, 1]
                    )
                    > 0
                ):
                    return col
        return None

    def find_block_3(self, board):
        opponent = P1 if self.player_token == P2 else P2
        for col in range(7):
            if Board.is_column_not_full(col, board):
                simulated_board = Board.simulate_move(col, opponent, board)
                if Board.evaluate_join_potential(simulated_board, opponent, [0, 1]) > 0:
                    return col
        return None

    def find_join_2(self, board):
        for col in range(7):
            if Board.is_column_not_full(col, board):
                simulated_board = Board.simulate_move(col, self.player_token, board)
                if (
                    Board.evaluate_join_potential(
                        simulated_board, self.player_token, [1, 0]
                    )
                    > 0
                ):
                    return col
        return None

    def find_random_move(self, board):
        valid_moves = Board.get_valid_moves(board)
        return random.choice(valid_moves) if valid_moves else None
