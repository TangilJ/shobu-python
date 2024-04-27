from enum import Enum, auto

from src import engine
from src.player import Player


class Winner(Enum):
    Player1 = auto()
    Player2 = auto()
    Loop = auto()


class Compare:
    def __init__(self, player1: Player, player2: Player):
        self.player1 = player1
        self.player2 = player2

    def _play(self) -> Winner:
        state = engine.start_state
        player = 1

        iters = 0
        while state.win == engine.Win.GameOngoing:
            if len(engine.get_next_states(state.board)) == 0:
                break

            if player == 1:
                state = self.player1.move(state)
            else:
                state = self.player2.move(state)
            player *= -1

            iters += 1
            if iters > 400:
                # If the game goes on for too long, it's a loop
                return Winner.Loop

        win_multiplier = 1 if state.win == engine.Win.OwnWin else -1
        return Winner.Player1 if player * win_multiplier == 1 else Winner.Player2

    def evaluate(self, num_games: int) -> (int, int, int):
        p1_wins = 0
        p2_wins = 0
        loops = 0

        for _ in range(num_games):
            winner = self._play()
            if winner == Winner.Player1:
                p1_wins += 1
            elif winner == Winner.Player2:
                p2_wins += 1
            else:
                loops += 1

        return p1_wins, p2_wins, loops


if __name__ == "__main__":
    from src.random_player import RandomPlayer

    GAMES = 10000

    p1 = RandomPlayer()
    p2 = RandomPlayer()
    print("Starting comparison")
    compare = Compare(p1, p2)
    p1_wins, p2_wins, loops = compare.evaluate(GAMES)
    print(f"Player 1 wins: {p1_wins}")
    print(f"Player 2 wins: {p2_wins}")
    print(f"Loops: {loops}")
