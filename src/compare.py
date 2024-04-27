from enum import Enum, auto

from src import engine
from src.mcts import AlphaZeroMCTS
from src.player import Player


class Winner(Enum):
    Player1 = auto()
    Player2 = auto()


class Compare:
    def __init__(self, player1: Player, player2: Player):
        self.player1 = player1
        self.player2 = player2

    def _play(self) -> Winner:
        state = engine.start_state
        player = 1

        while state.win == engine.Win.GameOngoing:
            if player == 1:
                state = self.player1.move(state)
            else:
                state = self.player2.move(state)
            player *= -1

        win_multiplier = 1 if state.win == engine.Win.OwnWin else -1
        return Winner.Player1 if player * win_multiplier == 1 else Winner.Player2

    def evaluate(self, num_games: int) -> float:
        p1_wins = 0
        for _ in range(num_games):
            winner = self._play()
            if winner == Winner.Player1:
                p1_wins += 1
        return p1_wins / num_games
