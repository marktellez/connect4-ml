import random

from src.constants import E0
from src.game.board import Board


class Game:
    def __init__(self, players):
        self.reset_board()
        self.players = players
        self.current_player = random.choice(players)

    def get_current_player(self):
        return self.current_player

    def get_opponent_player(self):
        return (
            self.players[0]
            if self.current_player == self.players[1]
            else self.players[1]
        )

    def reset_board(self):
        self.board = [[E0 for _ in range(7)] for _ in range(6)]
        self.move_history = []

    def drop_piece(self, column, player):
        open_row = -1
        for row in range(5, -1, -1):
            if self.board[row][column] == E0:
                open_row = row
                break

        if open_row == -1:
            return False

        self.board[open_row][column] = player
        self.move_history.append((player, row, column))
        self.current_player = (
            self.players[0]
            if self.players[1] == self.current_player
            else self.players[1]
        )

        return True

    def undo_move(self):
        if self.move_history:
            _, row, column = self.move_history.pop()
            self.board[row][column] = E0
            return True
        return False

    def is_game_over(self):
        if self.get_winner() is not None:
            return True

        if self.is_draw():
            return True

        return False

    def is_draw(self):
        return all(cell != E0 for row in self.board for cell in row)

    def get_winner(self):
        for check_method in [
            Board.find_horizontal_win,
            Board.find_vertical_win,
            Board.find_diagonal_right_win,
            Board.find_diagonal_left_win,
        ]:
            winning_sequence = check_method(self.board)
            if winning_sequence:
                # Check if all cells in the sequence match the same player token
                first_token = self.board[winning_sequence[0][0]][winning_sequence[0][1]]
                if all(
                    self.board[row][col] == first_token for row, col in winning_sequence
                ):
                    return first_token  # Return the winning player's token
        return None
