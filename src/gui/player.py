from constants import P1, P2, E0


class Player:
    def __init__(self, player_token):
        self.player_token = player_token
        self.opponent_token = P1 if player_token == P2 else P1
