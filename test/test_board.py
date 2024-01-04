import init
from game.board import Board
from constants import P1, P2, E0


class TestBoard:
    def setup_method(self):
        # Setup for each test, if needed
        self.empty_board = [[E0 for _ in range(7)] for _ in range(6)]
        # Add more setup as needed for different board states

    def test_score_board(self):
        # Example test for score_board
        score = Board.score_board(
            self.empty_board, P1, [10, 20], [5, 15], -30, [], [], []
        )
        assert score == 0  # Expecting 0 score for an empty board

    def test_get_all_sequences(self):
        # Example test for get_all_sequences
        sequences = Board.get_all_sequences(self.empty_board)
        expected_sequences = (
            (7 * 3) + (6 * 4) + (4 * 3 * 2)
        )  # vertical + horizontal + diagonal
        assert len(sequences) == expected_sequences

    def test_find_winning_combo_no_winner(self):
        # Example test for find_winning_combo with no winner
        winning_combo = Board.find_winning_combo(self.empty_board)
        assert winning_combo is None

    def test_find_horizontal_win(self):
        # Example test for find_horizontal_win
        # Setup a horizontal win
        board = self.empty_board
        board[0][:4] = [P1, P1, P1, P1]
        winning_combo = Board.find_horizontal_win(board)
        assert winning_combo == [(0, 0), (0, 1), (0, 2), (0, 3)]

    def test_find_vertical_win(self):
        # Example test for find_vertical_win
        # Setup a vertical win
        board = self.empty_board
        for i in range(4):
            board[i][0] = P1
        winning_combo = Board.find_vertical_win(board)
        assert winning_combo == [(0, 0), (1, 0), (2, 0), (3, 0)]

    def test_find_diagonal_right_win(self):
        # Test for diagonal right win
        board = self.empty_board
        for i in range(4):
            board[i][i] = P1  # Setting up a diagonal right win
        winning_combo = Board.find_diagonal_right_win(board)
        assert winning_combo == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_find_diagonal_left_win(self):
        # Test for diagonal left win
        board = self.empty_board
        for i in range(4):
            board[i][3 - i] = P1  # Setting up a diagonal left win
        winning_combo = Board.find_diagonal_left_win(board)
        expected_combo = [(0, 3), (1, 2), (2, 1), (3, 0)]
        assert sorted(winning_combo) == sorted(
            expected_combo
        )  # Compare without considering order

    def test_is_potential_join(self):
        # Test for potential join
        sequence = [P1, P1, E0, E0]
        assert Board.is_potential_join(sequence, P1)

    def test_is_potential_win(self):
        # Test for potential win
        sequence = [P1, P1, P1, P1]
        assert Board.is_potential_win(sequence, P1)

    def test_is_potential_block(self):
        # Test for potential block
        sequence = [P2, P2, E0, E0]
        assert Board.is_potential_block(sequence, P2)

    def test_is_potential_loss(self):
        # Test for potential loss
        sequence = [P2, P2, P2, E0]
        assert Board.is_potential_loss(sequence, P2)

    def test_simulate_move(self):
        # Test for simulate move
        board = self.empty_board
        new_board = Board.simulate_move(0, P1, board)
        assert new_board[5][0] == P1  # Expecting P1 at the bottom of the first column

    def test_is_column_not_full(self):
        # Test for checking if column is not full
        board = self.empty_board
        assert Board.is_column_not_full(0, board)

    def test_get_block_move(self):
        # Test for getting a block move
        board = self.empty_board
        # Set up a scenario where a block is needed
        board[5][:3] = [P2, P2, E0]
        block_move = Board.get_block_move(board, P1)
        assert block_move == 2  # Expecting a block in the third column

    def test_get_valid_moves(self):
        # Test for getting valid moves
        board = self.empty_board
        valid_moves = Board.get_valid_moves(board)
        assert len(valid_moves) == 7  # All columns should be valid in an empty board

    def test_horizontal_block_to_the_right(self):
        board = self.empty_board
        # P2 is about to win horizontally to the right
        board[5][0] = P2
        board[5][1] = P2
        board[5][2] = E0
        block_move = Board.get_block_move(board, P1)
        assert block_move == 2

    def test_horizontal_block_to_the_left(self):
        board = self.empty_board
        # P2 is about to win horizontally to the left
        board[5][1] = E0
        board[5][2] = P2
        board[5][3] = P2
        block_move = Board.get_block_move(board, P1)
        assert block_move == 1

    def test_vertical_block(self):
        board = self.empty_board
        for i in range(3):
            board[5 - i][0] = P2  # P2 is about to win vertically
        block_move = Board.get_block_move(board, P1)
        assert block_move == 0  # Expecting a block in the first column

    def test_diagonal_right_block(self):
        board = self.empty_board
        # Set up a board where P2 is about to win diagonally from bottom-right to top-left
        # and ensure there are no floating pieces
        board[5][3] = P2  # Bottom right
        board[4][2] = P2  # One up and to the left
        board[3][1] = P2  # Two up and to the left

        # Fill the spaces below the diagonal to avoid floating pieces
        board[5][2] = P1  # Filler
        board[4][1] = P1  # Filler
        board[5][1] = P1  # Filler
        board[5][0] = P1  # Filler

        block_move = Board.get_block_move(board, P1)
        assert block_move == 0

    def test_diagonal_left_block(self):
        board = self.empty_board
        # Set up a board where P2 is about to win diagonally from bottom-left to top-right
        # and ensure there are no floating pieces
        board[5][0] = P2  # Bottom left
        board[4][1] = P2  # One up and to the right
        board[3][2] = P2  # Two up and to the right

        # Fill the spaces below the diagonal to avoid floating pieces
        board[4][0] = P1  # Filler
        board[5][1] = P1  # Filler
        board[4][2] = P1  # Filler
        board[5][2] = P1  # Filler

        block_move = Board.get_block_move(board, P1)
        assert block_move == 3

    def test_is_potential_join(self):
        assert Board.is_potential_join([P1, P1, E0, E0], P1)  # 2 in a row
        assert Board.is_potential_join([P1, P1, P1, E0], P1)  # 3 in a row
        assert not Board.is_potential_join([P1, E0, E0, E0], P1)  # Only 1
        assert not Board.is_potential_join([P1, P2, P1, E0], P1)  # Interrupted sequence

    def test_is_potential_win(self):
        assert Board.is_potential_win([P1, P1, P1, P1], P1)  # 4 in a row
        assert not Board.is_potential_win([P1, P1, P1, E0], P1)  # Only 3
        assert not Board.is_potential_win([P1, P2, P1, P1], P1)  # Interrupted sequence

    def test_is_potential_block(self):
        assert Board.is_potential_block([P2, P2, E0, E0], P2)  # 2 opponent and 2 empty
        assert Board.is_potential_block([P2, P2, P2, E0], P2)  # 3 opponent and 1 empty
        assert not Board.is_potential_block([P1, P2, P2, E0], P2)  # Mixed players
        assert not Board.is_potential_block(
            [P2, P2, P2, P2], P2
        )  # All opponent, no empty

    def test_is_potential_loss(self):
        assert Board.is_potential_loss([P2, P2, P2, E0], P2)  # 3 opponent and 1 empty
        assert not Board.is_potential_loss([P2, P2, E0, E0], P2)  # Only 2 opponent
        assert not Board.is_potential_loss([P1, P2, P2, P2], P2)  # Mixed players

    def test_vertical_sequence(self):
        board = [[E0 for _ in range(7)] for _ in range(6)]
        board[5][0] = P1  # Assuming P1 is [1, 0, 0]
        board[4][0] = P1  # Assuming P1 is [1, 0, 0]
        board[3][0] = E0  # Assuming E0 is [0, 1, 0]

        print(f"board {board}")

        sequences = Board.get_relevant_sequences(board, 0)
        print("Generated Sequences:", sequences)  # Print the sequences here

        expected_sequence = [
            P1,
            P1,
            E0,
            E0,
        ]  # Adjusted format to match the actual output

        print("Expected Sequence:", expected_sequence)

        assert any(
            all(expected == cell for expected, cell in zip(expected_sequence, seq))
            for seq in sequences
        ), f"Expected sequence not found. Sequences: {sequences}"

    def test_horizontal_l_sequence(self):
        board = [[E0 for _ in range(7)] for _ in range(6)]
        board[5][0] = E0  # Assuming P1 is [1, 0, 0]
        board[5][1] = P1  # Assuming P1 is [1, 0, 0]
        board[5][2] = P1  # Assuming E0 is [0, 1, 0]

        print(f"board {board}")

        sequences = Board.get_relevant_sequences(board, 0)
        print("Generated Sequences:", sequences)  # Print the sequences here

        expected_sequence = [
            E0,
            E0,
            P1,
            P1,
        ]  # Adjusted format to match the actual output

        print("Expected Sequence:", expected_sequence)

        assert any(
            all(expected == cell for expected, cell in zip(expected_sequence, seq))
            for seq in sequences
        ), f"Expected sequence not found. Sequences: {sequences}"

    def test_horizontal_r_sequence(self):
        board = [[E0 for _ in range(7)] for _ in range(6)]
        board[5][0] = P1  # Assuming P1 is [1, 0, 0]
        board[5][1] = P1  # Assuming P1 is [1, 0, 0]
        board[5][2] = E0  # Assuming E0 is [0, 1, 0]

        print(f"board {board}")

        sequences = Board.get_relevant_sequences(board, 0)
        print("Generated Sequences:", sequences)  # Print the sequences here

        expected_sequence = [
            P1,
            P1,
            E0,
            E0,
        ]  # Adjusted format to match the actual output

        print("Expected Sequence:", expected_sequence)

        assert any(
            all(expected == cell for expected, cell in zip(expected_sequence, seq))
            for seq in sequences
        ), f"Expected sequence not found. Sequences: {sequences}"

    def test_diagonal_top_left_to_bottom_right_sequence(self):
        board = [[E0 for _ in range(7)] for _ in range(6)]
        board[5][0] = P1  # Assuming P1 is [1, 0, 0]
        board[4][1] = P1  # Assuming P1 is [1, 0, 0]
        board[3][2] = E0  # Assuming E0 is [0, 1, 0]

        print(f"board {board}")

        sequences = Board.get_relevant_sequences(board, 0)
        print("Generated Sequences:", sequences)  # Print the sequences here

        expected_sequence = [
            P1,
            P1,
            E0,
            E0,
        ]  # Adjusted format to match the actual output

        print("Expected Sequence:", expected_sequence)

        assert any(
            all(expected == cell for expected, cell in zip(expected_sequence, seq))
            for seq in sequences
        ), f"Expected sequence not found. Sequences: {sequences}"

    def test_diagonal_bottom_left_to_top_right_sequence(self):
        board = [[E0 for _ in range(7)] for _ in range(6)]
        board[2][0] = P1  # Assuming P1 is [1, 0, 0]
        board[3][1] = P1  # Assuming P1 is [1, 0, 0]
        board[4][2] = E0  # Assuming E0 is [0, 1, 0]

        print(f"board {board}")

        sequences = Board.get_relevant_sequences(board, 0)
        print("Generated Sequences:", sequences)  # Print the sequences here

        expected_sequence = [
            E0,
            P1,
            P1,
            E0,
        ]  # Adjusted format to match the actual output

        print("Expected Sequence:", expected_sequence)

        assert any(
            all(expected == cell for expected, cell in zip(expected_sequence, seq))
            for seq in sequences
        ), f"Expected sequence not found. Sequences: {sequences}"

    def test_find_winning_combo(self):
        board_state = [[E0 for _ in range(7)] for _ in range(6)]
        result = Board.find_winning_combo(board_state)
        # Assuming the function returns a combination of winning positions or None
        assert result is None or isinstance(result, list)

    def test_get_vertical_sequences(self):
        board_state = [[E0 for _ in range(7)] for _ in range(6)]
        result = Board.get_vertical_sequences(board_state)
        # Assuming the function returns a list of vertical sequences
        assert isinstance(result, list)

    def test_get_horizontal_sequences(self):
        board_state = [[E0 for _ in range(7)] for _ in range(6)]
        result = Board.get_horizontal_sequences(board_state)
        # Assuming the function returns a list of horizontal sequences
        assert isinstance(result, list)

    def test_get_horizontal_sequences_with_index(self):
        board_state = [[E0 for _ in range(7)] for _ in range(6)]
        result = Board.get_horizontal_sequences_with_index(board_state)
        # Assuming the function returns a list of horizontal sequences with indices
        assert isinstance(result, list)

    def test_get_diagonal_sequences(self):
        board_state = [[E0 for _ in range(7)] for _ in range(6)]
        result = Board.get_diagonal_sequences(board_state)
        # Assuming the function returns a list of diagonal sequences
        assert isinstance(result, list)

    def test_evaluate_join_potential(self):
        # Define valid test data for parameters
        test_board = [
            [E0 for _ in range(7)] for _ in range(6)
        ]  # Assuming E0 is the empty state
        test_player = P1  # Assuming P1 is a valid player identifier
        test_join_reward = 10  # Example reward value
        # Call the method and assert the result
        result = Board.evaluate_join_potential(
            test_board, test_player, test_join_reward
        )
        assert result is not None  # Adjust the assertion based on the expected output

    def test_score_sequence(self):
        # Define valid test data for parameters
        test_sequence = [
            E0,
            E0,
            E0,
            E0,
        ]  # Example sequence, assuming E0 is a valid state
        test_player = P1  # Assuming P1 is a valid player identifier
        test_join_reward = [10, 20]  # Example reward values
        test_block_reward = [5, 15]  # Example block reward values
        test_fail_to_block_win_penalty = -30  # Example penalty value
        test_join_success_counts = [0, 0]  # Example success counts
        test_block_success_counts = [0, 0]  # Example block success counts
        test_fail_to_block_win_counts = [0, 0]  # Example fail to block win counts

        # Call the method and assert the result
        result = Board.score_sequence(
            test_sequence,
            test_player,
            test_join_reward,
            test_block_reward,
            test_fail_to_block_win_penalty,
            test_join_success_counts,
            test_block_success_counts,
            test_fail_to_block_win_counts,
        )

        # Adjust the assertion based on the expected output
        # Example: assert result == expected_score
        assert isinstance(result, int)
