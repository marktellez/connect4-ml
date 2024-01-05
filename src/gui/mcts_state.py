import copy

from src.game.board import Board
from src.constants import P1, P2, E0


class MCTS4State:
    def __init__(self, game):
        self.rows = 6
        self.columns = 7
        self.win_length = 4
        self.current_player = game.get_current_player().player_token

        if game.board is None:
            raise Exception("Empty board - no no")
        else:
            self.board = copy.deepcopy(game.board)

        self.last_move = None

    def make_move(self, column):
        # Places a disc in the specified column
        old_board = copy.deepcopy(self.board)
        print(f"old_board {old_board}")
        for row in reversed(range(self.rows)):
            if self.board[row][column] == E0:  # Check if the slot is empty
                self.board[row][column] = self.current_player
                self.last_move = (row, column)
                self.current_player = P2 if self.current_player == P1 else P1
                break

        return old_board != self.board

    def get_valid_moves(self):
        return Board.get_valid_moves(self.board)

    def is_game_over(self):
        # Check if there's a winning combo or no valid moves left
        if Board.find_winning_combo(self.board) or not self.get_valid_moves():
            return True
        return False

    def get_winner(self):
        # Determine the winner based on the winning combo
        winning_combo = Board.find_winning_combo(self.board)
        if not winning_combo:
            return None
        # Get the player at the winning combo's first position
        return self.board[winning_combo[0][0]][winning_combo[0][1]]
