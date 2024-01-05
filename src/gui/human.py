import pygame

from src.gui.player import Player


class HumanPlayer(Player):
    def __init__(self, player_token, SQUARE_SIZE):
        super().__init__(player_token)
        self.SQUARE_SIZE = SQUARE_SIZE

    def move(self, pos):
        column = pos[0] // self.SQUARE_SIZE

        return column
