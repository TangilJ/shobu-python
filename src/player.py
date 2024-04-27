from abc import ABC, abstractmethod
from src import engine


class Player(ABC):
    @abstractmethod
    def move(self, state: engine.State) -> engine.State:
        pass
