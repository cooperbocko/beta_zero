from enum import Enum
from abc import ABC, abstractmethod

# player 1 is always 0 and player 2 is always 1
class Turn(int, Enum):
    PLAYER_1 = 0
    PLAYER_2 = 1

# player 1 win is always 0 and player 2 win is always 1 and draw is always 2
class Outcome(int, Enum):
    WIN_1 = 0
    WIN_2 = 1
    DRAW = 2
    
class Game(ABC):
    # returns a copy of the game
    @abstractmethod
    def copy(self):
        pass

    # completes a step of the game
    @abstractmethod
    def step(self, action: int):
        pass

    # returns the state of the game to be transformed into a tensor for your model
    @abstractmethod
    def get_state(self):
        pass

    # returns a list of valid moves with 1 being valid and 0 being invalid
    @abstractmethod
    def get_valid_moves(self) -> list[int]:
        pass

    # returns a unique state of the game for caching
    @abstractmethod
    def get_cache_state(self):
        pass

    # prints the board
    @abstractmethod
    def print_board(self):
        pass