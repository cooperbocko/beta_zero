import numpy as np

from game import Game
from mcts import MCTSNode

class C4MCTSNode(MCTSNode):
    def __init__(self, parent, state: Game, prior_prob: float):
        super().__init__(parent, state, prior_prob)
        
    def get_model_state(self):
        return self.state.get_state()