import pytest


from src.model.augment_game_data import GameDataAugmentation


def test_get_unique_moves():
    # Instantiate the GameDataAugmentation class
    augmenter = GameDataAugmentation()

    moves = [
        {
            "board_state": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "current_player": [1, 0, 0],
            "game_state": [1, 0, 0, 0],
            "best_move": [0, 0, 0, 0, 0, 0, 1],
            "valid_moves": [0, 1, 0, 0, 1, 1, 1],
        },
        {
            "board_state": [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            "current_player": [0, 1, 0],
            "game_state": [0, 1, 0, 0],
            "best_move": [0, 0, 0, 0, 0, 0, 1],
            "valid_moves": [0, 1, 0, 0, 1, 1, 1],
        },
        {
            "board_state": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "current_player": [1, 0, 0],
            "game_state": [1, 0, 0, 0],
            "best_move": [0, 0, 0, 0, 0, 0, 1],
            "valid_moves": [0, 1, 0, 0, 1, 1, 1],
        },  # Duplicate
        {
            "board_state": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            "current_player": [0, 0, 1],
            "game_state": [0, 0, 1, 0],
            "best_move": [0, 0, 0, 0, 0, 0, 1],
            "valid_moves": [0, 1, 0, 0, 1, 1, 1],
        },
    ]

    assert [
        {
            "board_state": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "current_player": [1, 0, 0],
            "game_state": [1, 0, 0, 0],
            "best_move": [0, 0, 0, 0, 0, 0, 1],
            "valid_moves": [0, 1, 0, 0, 1, 1, 1],
        },
        {
            "board_state": [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            "current_player": [0, 1, 0],
            "game_state": [0, 1, 0, 0],
            "best_move": [0, 0, 0, 0, 0, 0, 1],
            "valid_moves": [0, 1, 0, 0, 1, 1, 1],
        },
        {
            "board_state": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            "current_player": [0, 0, 1],
            "game_state": [0, 0, 1, 0],
            "best_move": [0, 0, 0, 0, 0, 0, 1],
            "valid_moves": [0, 1, 0, 0, 1, 1, 1],
        },
    ] == augmenter.get_unique_moves(moves)


def test_process_moves():
    augmenter = GameDataAugmentation()
    moves = [
        {
            "board_state": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "current_player": [1, 0, 0],
            "game_state": [1, 0, 0, 0],
            "best_move": [0, 0, 0, 0, 0, 0, 1],
            "valid_moves": [0, 1, 0, 0, 1, 1, 1],
        },
        # Duplicate move
        {
            "board_state": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "current_player": [1, 0, 0],
            "game_state": [1, 0, 0, 0],
            "best_move": [0, 0, 0, 0, 0, 0, 1],
            "valid_moves": [0, 1, 0, 0, 1, 1, 1],
        },
        {
            "board_state": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            "current_player": [0, 0, 1],
            "game_state": [0, 0, 1, 0],
            "best_move": [0, 0, 0, 0, 0, 0, 1],
            "valid_moves": [0, 1, 0, 0, 1, 1, 1],
        },
    ]

    processed_moves = augmenter.process_moves(moves)

    assert len(processed_moves) == 2


def test_flip_board_state():
    # Sample input move dictionary
    input_move = {
        "board_state": [
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        ],
        "current_player": [1, 0, 0],
        "best_move": [0, 0, 0, 1],
        "game_state": [0, 0, 1, 0],
        "valid_moves": [0, 1, 0, 0, 1, 1, 1],
    }

    # Expected output for the flipped board state
    expected_flipped = {
        "board_state": [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        ],
        "current_player": [1, 0, 0],
        "best_move": [1, 0, 0, 0],
        "game_state": [0, 0, 1, 0],
        "valid_moves": [1, 1, 1, 0, 0, 1, 0],
    }

    # Instantiate the class and call the method to flip the board state
    augmenter = GameDataAugmentation()
    flipped_move = augmenter.flip_board_state(input_move)

    # Assert that the flipped board state matches the expected output
    assert flipped_move == expected_flipped


def test_mirror_board_state():
    # Sample input move dictionary
    input_move = {
        "board_state": [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        ],
        "current_player": [1, 0, 0],
        "best_move": [0, 0, 0, 1],
        "game_state": [1, 0, 0, 0],
        "valid_moves": [0, 1, 0, 0, 1, 1, 1],
    }

    # Expected output for the mirrored board state
    expected_mirrored = {
        "board_state": [
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ],
        "current_player": [0, 1, 0],
        "best_move": [0, 0, 0, 1],
        "game_state": [0, 1, 0, 0],
        "valid_moves": [0, 1, 0, 0, 1, 1, 1],
    }

    # Instantiate the class and call the method to mirror the board state
    augmenter = GameDataAugmentation()
    mirrored_move = augmenter.mirror_board_state(input_move)

    # Assert that the mirrored board state matches the expected output
    assert mirrored_move == expected_mirrored
