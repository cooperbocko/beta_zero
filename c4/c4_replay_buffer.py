import random

import numpy as np


class C4ReplayBuffer():
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        
    def add(self, states, action_probs, values):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((states, action_probs, values))
        
    def sample(self, batch_size):
        states = []
        action_probs = []
        values = []
        game_lengths = [len(game[0]) for game in self.buffer]
        sampled_games = random.choices(self.buffer, weights=game_lengths, k=batch_size)
        
        for game in sampled_games:
            start = random.randint(0, len(game[0]) - 1)
            
            curr_state = game[0][start]
            curr_action_probs = game[1][start]
            curr_value = game[2][start]
            
            player = curr_state[0]
            opponent = curr_state[1]
            
            state = np.stack((player, opponent))
            action_probs.append(curr_action_probs)
            values.append(curr_value)
            states.append(state)
            
        return states, action_probs, values