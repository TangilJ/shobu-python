import random

from src.player import Player
from src import engine


class RandomPlayer(Player):
    def move(self, state: engine.State) -> engine.State:
        next_states = engine.get_next_states(state.board)
        return random.choice(next_states)
