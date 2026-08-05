import numpy as np

from game import Game

class C4MCTSNode:
    def __init__(self, parent, state: Game, prior_prob: float):
        self.parent = parent
        self.state = state
        self.children = {}
        self.visits = 0
        self.t_action_value = 0
        self.m_action_value = 0
        self.prior_prob = prior_prob
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def create_node(self, parent, state, prior_prob):
        return type(self)(parent, state, prior_prob)
    
    def get_model_state(self):
        curr_state = self.state.get_state()            
        player = curr_state[0]
        opponent = curr_state[1]
        model_state = np.stack((player, opponent))
        return model_state