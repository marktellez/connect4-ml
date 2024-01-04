import init
from game.game import Game
from constants import P1, P2, E0


class TestGame:
    def setup_method(self):
        self.game = Game([P1, P2])

    def test_board_initialization(self):
        game = Game([P1, P2])
        expected_rows = 6
        expected_columns = 7
        assert len(game.board) == expected_rows
        for row in game.board:
            assert len(row) == expected_columns
            assert all(cell == E0 for cell in row)

    def test_drop_piece_empty_column(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
        ]
        column = 0
        assert self.game.drop_piece(column, P1)
        assert self.game.board[5][column] == P1

    def test_drop_piece_partially_filled_column(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, P1, E0, E0, E0, E0, E0],
        ]
        column = 1
        assert self.game.drop_piece(column, P2)
        assert self.game.board[4][column] == P2

    def test_drop_piece_full_column(self):
        self.game.board = [
            [P2, E0, E0, E0, E0, E0, E0],
            [P2, E0, E0, E0, E0, E0, E0],
            [P2, E0, E0, E0, E0, E0, E0],
            [P2, E0, E0, E0, E0, E0, E0],
            [P2, E0, E0, E0, E0, E0, E0],
            [P2, E0, E0, E0, E0, E0, E0],
        ]
        column = 0
        assert not self.game.drop_piece(column, P1)

    def test_board_state_after_dropping_piece(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
        ]
        column = 3
        self.game.drop_piece(column, P1)
        assert self.game.board[5][column] == P1

    def test_move_history_after_dropping_piece(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
        ]
        column = 4
        self.game.drop_piece(column, P2)
        last_move = self.game.move_history[-1]
        assert last_move == (P2, 5, column)

    def test_draw_board(self):
        self.game.board = [
            [P2, P2, P1, P2, P2, P1, P2],
            [P2, P1, P2, P1, P1, P2, P1],
            [P2, P1, P1, P2, P1, P2, P1],
            [P1, P2, P2, P1, P2, P1, P2],
            [P2, P1, P2, P2, P2, P1, P1],
            [P1, P2, P1, P1, P2, P1, P2],
        ]
        assert self.game.is_draw(), "No winner - draw board."

    def test_horizontal_winner_1(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [P1, P1, P1, P1, E0, E0, E0],
        ]
        assert (
            self.game.get_winner() == P1
        ), "Player 1 should be the winner horizontally."

    def test_horizontal_winner_2(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, P1, P1, P1, P1, E0, E0],
            [P1, P1, P1, P2, P2, E0, E0],
        ]
        assert (
            self.game.get_winner() == P1
        ), "Player 1 should be the winner horizontally."

    def test_horizontal_winner_3(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [P1, P1, P1, P1, P2, E0, E0],
            [P2, P1, P1, P2, P1, E0, E0],
            [P1, P1, P1, P2, P2, E0, E0],
        ]
        assert (
            self.game.get_winner() == P1
        ), "Player 1 should be the winner horizontally."

    def test_no_horizontal_winner(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [P1, P1, P2, P1, P2, E0, E0],
            [P2, P1, P1, P2, P1, E0, E0],
            [P1, P1, P1, P2, P2, E0, E0],
        ]
        assert self.game.get_winner() is None, "No winner horizontally."

    def test_vertical_winner_1(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, E0, E0, E0, E0, E0],
            [P1, E0, E0, E0, E0, E0, E0],
            [P1, E0, E0, E0, E0, E0, E0],
            [P1, E0, E0, E0, E0, E0, E0],
            [P1, E0, E0, E0, E0, E0, E0],
        ]
        assert self.game.get_winner() == P1, "Player 1 should be the winner vertically."

    def test_vertical_winner_2(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, P1, E0, E0, E0, E0],
            [E0, E0, P1, E0, E0, E0, E0],
            [P1, E0, P1, E0, E0, E0, E0],
            [P1, E0, P1, E0, E0, E0, E0],
            [P1, E0, P2, E0, E0, E0, E0],
        ]
        assert self.game.get_winner() == P1, "Player 1 should be the winner vertically."

    def test_vertical_winner_3(self):
        self.game.board = [
            [E0, E0, E0, P1, E0, E0, E0],
            [E0, E0, P2, P1, E0, E0, E0],
            [E0, E0, P1, P1, E0, E0, E0],
            [P1, E0, P1, P1, E0, E0, E0],
            [P1, E0, P1, P2, E0, E0, E0],
            [P1, E0, P2, P2, E0, E0, E0],
        ]
        assert self.game.get_winner() == P1, "Player 1 should be the winner vertically."

    def test_no_vertical_winner(self):
        self.game.board = [
            [E0, E0, E0, P2, E0, E0, E0],
            [E0, E0, P2, P1, E0, E0, E0],
            [E0, E0, P1, P1, E0, E0, E0],
            [P1, E0, P1, P1, E0, E0, E0],
            [P1, E0, P1, P2, E0, E0, E0],
            [P1, E0, P2, P2, E0, E0, E0],
        ]
        assert self.game.get_winner() is None, "No winner vertically."

    def test_diagonal_winner_1(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, P2, P2, E0, E0, E0],
            [E0, E0, P2, P1, E0, E0, E0],
            [P1, P2, P1, P1, E0, E0, E0],
            [P2, P2, P1, P2, E0, E0, E0],
            [P1, P1, P2, P2, E0, E0, E0],
        ]
        assert (
            self.game.get_winner() == P2
        ), "Player 2 should be the winner diagonally right."

    def test_no_diagonal_winner_r(self):
        self.game.board = [
            [E0, E0, E0, E0, E0, E0, E0],
            [E0, E0, P2, P2, E0, E0, E0],
            [E0, E0, P1, P1, E0, E0, E0],
            [P1, P2, P1, P1, E0, E0, E0],
            [P2, P2, P1, P2, E0, E0, E0],
            [P1, P1, P2, P2, E0, E0, E0],
        ]
        assert self.game.get_winner() is None, "No winner diagonally right."

    def test_diagonal_winner_2(self):
        self.game.board = [
            [P2, E0, E0, E0, E0, E0, E0],
            [P1, P2, P2, P2, E0, E0, E0],
            [P2, P1, P2, P1, E0, E0, E0],
            [P1, P2, P1, P2, E0, E0, E0],
            [P2, P2, P1, P2, P1, E0, E0],
            [P1, P1, P2, P2, P2, E0, E0],
        ]
        assert (
            self.game.get_winner() == P2
        ), "Player 2 should be the winner diagonally left."

    def test_no_diagonal_winner_l(self):
        self.game.board = [
            [P1, E0, E0, E0, E0, E0, E0],
            [P1, P2, P2, P1, E0, E0, E0],
            [P2, P1, P2, P1, E0, E0, E0],
            [P1, P2, P1, P2, E0, E0, E0],
            [P2, P2, P1, P2, P1, E0, E0],
            [P1, P1, P2, P2, P2, E0, E0],
        ]
        assert self.game.get_winner() is None, "No winner diagonally left."

    def test_undo_move(self):
        self.game = Game([P1, P2])
        self.game.drop_piece(0, P1)  # Player 1
        self.game.drop_piece(1, P2)  # Player 2

        # Undo the last move (Player 2's move)
        self.game.undo_move()
        assert (
            self.game.board[5][1] == E0
        ), "Last move should be undone (cell should be empty)."

        # Check if the second last move (Player 1's move) is still there
        assert self.game.board[5][0] == P1, "Second last move should still be present."

        # Undo the second last move (Player 1's move)
        self.game.undo_move()
        assert (
            self.game.board[5][0] == E0
        ), "Second last move should be undone (cell should be empty)."
