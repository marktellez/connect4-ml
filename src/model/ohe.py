from src.constants import P1, P2


def move_to_ohe(column_index):
    return [1 if i == column_index else 0 for i in range(7)]


def valid_moves_to_ohe(valid_moves):
    return [1 if i in valid_moves else 0 for i in range(7)]


def winner_to_ohe(winner=None, is_draw=False):
    if winner == P1:  # Player 1 wins
        return [
            1,
            0,
            0,
            0,
        ]
    elif winner == P2:  # Player 2 wins
        return [0, 1, 0, 0]
    elif is_draw:  # It's a draw
        return [0, 0, 1, 0]
    else:  # Game is still ongoing
        return [0, 0, 0, 1]
