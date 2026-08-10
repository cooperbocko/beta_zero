from enum import Enum
from abc import ABC, abstractmethod

class Turn(int, Enum):
    PLAYER_1 = 0
    PLAYER_2 = 1
    
class Outcome(int, Enum):
    WIN_1 = 0
    WIN_2 = 1
    DRAW = 2
    
class Game(ABC):
    @abstractmethod
    def copy(self):
        pass
        
    @abstractmethod
    def step(self, action: int):
        pass
        
    @abstractmethod
    def get_state(self):
        pass
        
    @abstractmethod
    def get_valid_moves(self) -> list[int]:
        pass
    
    @abstractmethod
    def get_cache_state(self):
        pass