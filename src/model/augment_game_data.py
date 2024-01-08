import random
from tqdm import tqdm
import os
import glob
import json
import hashlib


class GameDataAugmentation:
    def flip_board_state(self, move):
        # Flip the board state left to right
        flipped_board = [row[::-1] for row in move["board_state"]]
        flipped_valid_moves = move["valid_moves"][::-1]
        flipped_best_move = move["best_move"][::-1]

        # Return the flipped move dictionary
        return {
            "board_state": flipped_board,
            "current_player": move["current_player"],
            "game_state": move["game_state"],
            "best_move": flipped_best_move,
            "valid_moves": flipped_valid_moves,
        }

    def mirror_board_state(self, move):
        # Helper function to mirror a single state
        def mirror_state(state):
            if state == [1, 0, 0]:
                return [0, 1, 0]
            elif state == [0, 1, 0]:
                return [1, 0, 0]
            else:
                return state

        # Mirror the board state and current player
        mirrored_board = [
            [mirror_state(cell) for cell in row] for row in move["board_state"]
        ]
        mirrored_current_player = mirror_state(move["current_player"])

        # Mirror the game state
        if move["game_state"][:2] == [1, 0]:
            mirrored_game_state = [0, 1] + move["game_state"][2:]
        elif move["game_state"][:2] == [0, 1]:
            mirrored_game_state = [1, 0] + move["game_state"][2:]
        else:
            mirrored_game_state = move["game_state"]

        # Return the mirrored move dictionary
        return {
            "board_state": mirrored_board,
            "current_player": mirrored_current_player,
            "game_state": mirrored_game_state,
            "best_move": move["best_move"],
            "valid_moves": move["valid_moves"],
        }

    def get_unique_moves(self, moves):
        unique_moves = {}

        for move in moves:
            move_hash = hashlib.md5(
                json.dumps(move, sort_keys=True).encode()
            ).hexdigest()

            if move_hash not in unique_moves:
                unique_moves[move_hash] = move

        return list(unique_moves.values())

    def process_moves(self, game_states):
        unique_moves = self.get_unique_moves(game_states)
        processed_moves = []
        for move in unique_moves:
            flipped = self.flip_board_state(move)
            mirrored = self.mirror_board_state(flipped)
            processed_moves.append(mirrored)
        return processed_moves

    def balance_terminal_moves(self, moves):
        terminal_moves = []
        non_terminal_moves = []

        # Split the moves into terminal and non-terminal
        for move in moves:
            if move["game_state"][0] == 1 or move["game_state"][1] == 1:
                terminal_moves.append(move)
            else:
                non_terminal_moves.append(move)

        print(f"terminal_moves {len(terminal_moves)}")
        print(f"non_terminal_moves {len(non_terminal_moves)}")

        # Balance the dataset
        min_count = min(len(terminal_moves), len(non_terminal_moves))
        balanced_moves = terminal_moves[:min_count] + non_terminal_moves[:min_count]

        # Optionally shuffle the balanced dataset
        random.shuffle(balanced_moves)

        return balanced_moves


def main():
    augmenter = GameDataAugmentation()
    game_state_files = glob.glob("game_states/*.json")

    # Remove existing files in the training_data directory
    [os.remove(f) for f in glob.glob("/training_data/*")]

    all_moves = []
    # Load all moves from all files
    for file in game_state_files:
        with open(file, "r") as f:
            moves = json.load(f)
        all_moves.extend(moves)

    # Process and get unique moves
    processed_moves = augmenter.process_moves(all_moves)
    unique_moves = augmenter.get_unique_moves(processed_moves)

    # Balance terminal and non-terminal moves
    balanced_moves = augmenter.balance_terminal_moves(unique_moves)

    # Calculate total batches
    total_batches = len(balanced_moves) // 100

    # Counter for the output file names
    file_counter = 1

    with tqdm(total=total_batches, desc="Overall Progress") as pbar:
        # Splitting into batches of 100 and writing to training_data files
        for i in range(0, len(balanced_moves), 100):
            batch = balanced_moves[i : i + 100]
            batch_file = f"./training_data/{file_counter}.json"
            with open(batch_file, "w") as f:
                json.dump(batch, f, indent=4)

            file_counter += 1
            pbar.update(1)

    print(f"Moves written: {file_counter * 100}")


if __name__ == "__main__":
    main()
