import numpy as np

from game import Game
from mcts import MCTSNode

class C4MCTSNode(MCTSNode):
    def __init__(self, parent, state: Game, prior_prob: float):
        super().__init__(parent, state, prior_prob)
        
    def get_model_state(self):
        curr_state = self.state.get_state()            
        player = curr_state[0]
        opponent = curr_state[1]
        model_state = np.stack((player, opponent))
        return model_state