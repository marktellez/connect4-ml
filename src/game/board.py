import copy
from src.constants import P1, P2, E0


class Board:
    @staticmethod
    def score_board(
        board,
        player,
        join_reward,
        block_reward,
        fail_to_block_win_penalty,
        join_success_counts,
        block_success_counts,
        fail_to_block_win_counts,
    ):
        total_score = 0
        for sequence in Board.get_all_sequences(board):
            total_score += Board.score_sequence(
                sequence,
                player,
                join_reward,
                block_reward,
                fail_to_block_win_penalty,
                join_success_counts,
                block_success_counts,
                fail_to_block_win_counts,
            )
        return total_score

    @staticmethod
    def get_all_sequences(board):
        """
        Get all possible sequences of 4 in the board state that can lead to a join.

        :param board: The board state.
        :return: A list of all sequences.
        """
        return (
            Board.get_vertical_sequences(board)
            + Board.get_horizontal_sequences(board)
            + Board.get_diagonal_sequences(board)
        )

    @staticmethod
    def find_winning_combo(board):
        return (
            Board.find_horizontal_win(board)
            or Board.find_vertical_win(board)
            or Board.find_diagonal_right_win(board)
            or Board.find_diagonal_left_win(board)
        )

    @staticmethod
    def find_horizontal_win(board):
        for row in range(6):
            for col in range(7 - 3):
                if board[row][col] != E0 and all(
                    board[row][col] == board[row][col + i] for i in range(1, 4)
                ):
                    return [(row, col + i) for i in range(4)]
        return None

    @staticmethod
    def find_vertical_win(board):
        for col in range(7):
            for row in range(6 - 3):
                if board[row][col] != E0 and all(
                    board[row][col] == board[row + i][col] for i in range(1, 4)
                ):
                    return [(row + i, col) for i in range(4)]
        return None

    @staticmethod
    def find_diagonal_right_win(board):
        for row in range(6 - 3):
            for col in range(7 - 3):
                if board[row][col] != E0 and all(
                    board[row + i][col + i] == board[row][col] for i in range(1, 4)
                ):
                    return [(row + i, col + i) for i in range(4)]
        return None

    @staticmethod
    def find_diagonal_left_win(board):
        for row in range(3, 6):
            for col in range(7 - 3):
                if board[row][col] != E0 and all(
                    board[row - i][col + i] == board[row][col] for i in range(1, 4)
                ):
                    return [(row - i, col + i) for i in range(4)]
        return None

    @staticmethod
    def get_vertical_sequences(board):
        sequences = []
        for col in range(7):
            for row in range(6 - 3):  # Check all vertical sequences of length 4
                sequence = [board[row + i][col] for i in range(4)]
                sequences.append(sequence)
        return sequences

    @staticmethod
    def get_horizontal_sequences(board):
        sequences = []
        for row in range(6):
            for col in range(4):
                sequences.append(board[row][col : col + 4])
        return sequences

    @staticmethod
    def get_horizontal_sequences_with_index(board):
        sequences_with_index = []
        for row in range(6):
            for col in range(4):  # 4 sequences per row
                sequence = board[row][col : col + 4]
                sequences_with_index.append(
                    (sequence, col)
                )  # Include starting column index
        return sequences_with_index

    @staticmethod
    def get_diagonal_sequences(board):
        sequences = []
        # Diagonal from top-left to bottom-right
        for row in range(3):
            for col in range(4):
                sequences.append([board[row + i][col + i] for i in range(4)])
        # Diagonal from bottom-left to top-right
        for row in range(3, 6):
            for col in range(4):
                sequences.append([board[row - i][col + i] for i in range(4)])
        return sequences

    @staticmethod
    def evaluate_join_potential(board, player, join_reward):
        score = 0
        for sequence in Board.get_all_sequences(board):
            if Board.is_potential_join(sequence, player):
                if sequence.count(player) == 2:
                    score += join_reward[0]
                elif sequence.count(player) == 3:
                    score += join_reward[1]
        return score

    @staticmethod
    def score_sequence(
        sequence,
        player,
        join_reward,
        block_reward,
        fail_to_block_win_penalty,
        join_success_counts,
        block_success_counts,
        fail_to_block_win_counts,
    ):
        opponent = P1 if player == P2 else P2
        if Board.is_potential_join(sequence, player):
            if sequence.count(player) == 3:
                join_success_counts.append(1)
                return join_reward[1]

            elif sequence.count(player) == 2:
                join_success_counts.append(1)
                return join_reward[0]

        if Board.is_potential_block(sequence, opponent):
            if sequence.count(opponent) == 3:
                block_success_counts.append(1)
                return block_reward[1]
            elif sequence.count(opponent) == 2:
                block_success_counts.append(1)
                return block_reward[0]

        if sequence.count(opponent) == 3 and sequence.count(E0) == 1:
            fail_to_block_win_counts.append(1)
            return fail_to_block_win_penalty

        return 0

    @staticmethod
    def is_potential_join(sequence, player):
        for i in range(len(sequence)):
            if sequence[i] == player:
                count = 1
                for j in range(i + 1, min(i + 3, len(sequence))):
                    if sequence[j] != player:
                        break
                    count += 1
                if count in [2, 3]:
                    return True
        return False

    @staticmethod
    def is_potential_win(sequence, player):
        for i in range(len(sequence)):
            if sequence[i] == player:
                count = 1
                for j in range(i + 1, min(i + 4, len(sequence))):
                    if sequence[j] != player:
                        break
                    count += 1
                if count in [4]:
                    return True
        return False

    @staticmethod
    def is_potential_win_board(board, player):
        # Check for potential wins in all sequences on the board
        for sequence in Board.get_all_sequences(board):
            if Board.is_potential_win(sequence, player):
                return True
        return False

    @staticmethod
    def is_potential_block(sequence, opponent):
        opponent_count = sequence.count(opponent)
        empty_count = sequence.count(E0)
        return (opponent_count in [2, 3]) and empty_count == (4 - opponent_count)

    @staticmethod
    def is_potential_loss(sequence, opponent):
        opponent_count = sequence.count(opponent)
        empty_count = sequence.count(E0)
        return opponent_count == 3 and empty_count == 1

    @staticmethod
    def simulate_move(col, player, board):
        new_board = copy.deepcopy(board)
        for row in range(5, -1, -1):
            if new_board[row][col] == E0:
                new_board[row][col] = player
                break
        return new_board

    @staticmethod
    def is_column_not_full(col, board):
        return board[0][col] == E0

    @staticmethod
    def get_block_move(board, current_player):
        opponent = P1 if current_player == P2 else P2
        for sequence, indices in Board.get_all_sequences_with_indices(board):
            if Board.is_potential_block(sequence, opponent):
                # Look for the empty spot in the sequence that's closest to the opponent's tokens
                for idx, (row, col) in enumerate(indices):
                    if sequence[idx] == E0 and Board.is_column_not_full(col, board):
                        # Check if this block is more immediate than others
                        if (idx > 0 and sequence[idx - 1] == opponent) or (
                            idx < len(sequence) - 1 and sequence[idx + 1] == opponent
                        ):
                            return col
        return None

    @staticmethod
    def get_all_sequences_with_indices(board):
        return (
            Board.get_vertical_sequences_with_index(board)
            + Board.get_horizontal_sequences_with_index(board)
            + Board.get_diagonal_sequences_with_index(board)
        )

    @staticmethod
    def get_horizontal_sequences_with_index(board):
        sequences_with_indices = []
        for row in range(6):
            for col in range(7 - 3):  # Only rightward sequences
                sequence = [board[row][col + i] for i in range(4)]
                indices = [(row, col + i) for i in range(4)]
                sequences_with_indices.append((sequence, indices))
        return sequences_with_indices

    @staticmethod
    def get_vertical_sequences_with_index(board):
        sequences_with_indices = []
        for col in range(7):
            for row in range(6 - 3):
                sequence = [board[row + i][col] for i in range(4)]
                indices = [(row + i, col) for i in range(4)]
                sequences_with_indices.append((sequence, indices))
        return sequences_with_indices

    @staticmethod
    def get_diagonal_sequences_with_index(board):
        sequences_with_indices = []
        # Diagonal from top-left to bottom-right
        for row in range(3):
            for col in range(4):
                sequence = [board[row + i][col + i] for i in range(4)]
                indices = [(row + i, col + i) for i in range(4)]
                sequences_with_indices.append((sequence, indices))

        # Diagonal from bottom-left to top-right
        for row in range(3, 6):
            for col in range(4):
                sequence = [board[row - i][col + i] for i in range(4)]
                indices = [(row - i, col + i) for i in range(4)]
                sequences_with_indices.append((sequence, indices))

        return sequences_with_indices

    @staticmethod
    def get_valid_moves(board):
        valid_moves = []

        for col in range(7):
            for row in range(5, -1, -1):
                if board[row][col] == E0:
                    valid_moves.append(col)

        return list(set(valid_moves))

    @staticmethod
    def is_column_feasible_for_vertical_win(col, board):
        # Count empty spaces in the column
        empty_spaces = sum(1 for row in board if row[col] == E0)
        return empty_spaces >= 4  # At least 4 spaces needed for a vertical win

    @staticmethod
    def get_relevant_sequences(board, col):
        # Find the row where the piece would land
        row = next((r for r in range(5, -1, -1) if board[r][col] == E0), None)
        if row is None:  # Column is full, no sequences to check
            return []

        relevant_sequences = []

        # Vertical sequences
        for i in range(4):
            if row + i <= 5:  # Check if row index is within bounds
                sequence = [board[row + i][col]]
                relevant_sequences.append(sequence)

        # Horizontal sequences
        for start_col in range(max(0, col - 3), min(7, col + 4)):
            if start_col + 3 < 7:  # Check that the end of the sequence is within bounds
                sequence = [board[row][start_col + i] for i in range(4)]
                relevant_sequences.append(sequence)

        # Diagonal sequences (top-left to bottom-right)
        for row_offset in range(-3, 1):
            diag_row = row + row_offset
            diag_col = col + row_offset
            if 0 <= diag_row <= 2 and 0 <= diag_col <= 3:
                sequence = [board[diag_row + i][diag_col + i] for i in range(4)]
                relevant_sequences.append(sequence)

        # Diagonal sequences (bottom-left to top-right)
        for row_offset in range(-3, 1):
            diag_row = row - row_offset
            diag_col = col + row_offset
            if 3 <= diag_row <= 5 and 0 <= diag_col <= 3:
                sequence = [board[diag_row - i][diag_col + i] for i in range(4)]
                relevant_sequences.append(sequence)

        return relevant_sequences

    @staticmethod
    def count_played_pieces(board):
        count = 0
        for row in board:
            for cell in row:
                if cell == P1 or cell == P2:
                    count += 1
        return count

    @staticmethod
    def analyze_board_post_move(board, player_token):
        missed_win = Board.find_missed_win_opportunity(board, player_token)
        missed_block = Board.find_missed_block_opportunity(board, player_token)

        return missed_win, missed_block
