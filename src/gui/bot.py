import random
import math
import copy

from src.gui.mcts_state import MCTS4State
from src.gui.player import Player


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.visits = 0
        self.reward = 0
        self.children = []
        self.untried_moves = state.get_valid_moves()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0


class BotPlayer(Player):
    def __init__(self, player_token):
        super().__init__(player_token)
        self.iterations = random.randint(100, 2000)
        self.exploration_weight = random.uniform(0.5, 2.0)

    def set_game(self, game):
        print(f"Setting game")
        self.game = game

    def move(self, board):
        root_state = MCTS4State(self.game)
        root_node = MCTSNode(root_state)
        print(f"root_state {root_state} root_node {root_node}")

        for _ in range(self.iterations):
            print(f"iterating")
            node = self.select_node(root_node)
            winner = self.simulate(node.state)
            print(f"winner {winner} node {node}")
            self.backpropagate(node, winner)

        print("finding best move")
        best_move = self.best_move(root_node)
        print(f"best_move {best_move}")
        return best_move

    def select_node(self, node):
        while not node.state.is_game_over():
            if not node.is_fully_expanded():
                return self.expand_node(node)
            else:
                node = self.uct_select(node)
        return node

    def expand_node(self, node):
        move = node.untried_moves.pop()
        new_state = copy.deepcopy(node.state)
        new_state.make_move(move)
        child_node = MCTSNode(new_state, parent=node)
        node.children.append(child_node)
        return child_node

    def simulate(self, state):
        while not state.is_game_over():
            print("not game over")
            possible_moves = state.get_valid_moves()
            print(f"possible_moves {possible_moves}")

            print(f"before move board {state.board}")
            move = random.choice(possible_moves)
            print(f"move {move}")
            if not state.make_move(move):
                raise Exception("No move registered!")

            print(f"after move board {state.board}")
        return state.get_winner()

    def backpropagate(self, node, winner):
        while node:
            node.visits += 1
            if node.state.current_player == winner:
                node.reward += 1
            node = node.parent

    def uct_select(self, node):
        best_value = float("-inf")
        best_node = None
        for child in node.children:
            exploit = child.reward / child.visits
            explore = math.sqrt(math.log(node.visits) / child.visits)
            value = exploit + self.exploration_weight * explore
            if value > best_value:
                best_value = value
                best_node = child
        return best_node

    def best_move(self, node):
        print(f"node {node}")
        print(f"node.children {node.children}")

        best_visits = max(child.visits for child in node.children)
        best_children = [
            child for child in node.children if child.visits == best_visits
        ]

        best_move_node = random.choice(best_children)
        _, best_move_column = best_move_node.state.last_move
        return best_move_column
