import pygame
from src.constants import P1, P2


class GUIBoard:
    def __init__(self, width, height, square_size):
        self.width = width
        self.height = height
        self.square_size = square_size
        self.radius = square_size // 2 - 5

        # Colors
        self.red = (255, 0, 0)
        self.yellow = (255, 255, 0)
        self.blue = (0, 0, 255)
        self.dkblue = (0, 0, 128)

    def draw(self, board, screen):
        screen.fill(self.blue)
        for c in range(7):
            for r in range(6):
                pygame.draw.circle(
                    screen,
                    self.dkblue,
                    (
                        int(c * self.square_size + self.square_size // 2),
                        int(r * self.square_size + self.square_size // 2),
                    ),
                    self.radius,
                )

                if board[r][c] == P1:
                    pygame.draw.circle(
                        screen,
                        self.red,
                        (
                            int(c * self.square_size + self.square_size // 2),
                            int(r * self.square_size + self.square_size // 2),
                        ),
                        self.radius,
                    )
                elif board[r][c] == P2:
                    pygame.draw.circle(
                        screen,
                        self.yellow,
                        (
                            int(c * self.square_size + self.square_size // 2),
                            int(r * self.square_size + self.square_size // 2),
                        ),
                        self.radius,
                    )
