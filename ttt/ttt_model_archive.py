import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import ResBlock

class TTTReplayBuffer():
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
                   
class TTTResNet(nn.Module):
    def __init__(self, input_channels=2, n_blocks=3, n_channels=32, value_layers=16, n_actions=9):
        super(TTTResNet, self).__init__()
        self.conv_init = nn.Conv2d(input_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.gn_init = nn.GroupNorm(num_groups=n_channels//8, num_channels=n_channels)
        
        self.res_blocks = nn.ModuleList([ResBlock(n_channels) for _ in range(n_blocks-1)])
        
        self.policy_conv = nn.Conv2d(n_channels, 2, kernel_size=1, bias=False)
        self.policy_gn = nn.GroupNorm(num_groups=1, num_channels=2)
        self.policy = nn.Linear(2*3*3, n_actions)
        
        self.value_conv = nn.Conv2d(n_channels, 1, kernel_size=1, bias=False)
        self.value_gn = nn.GroupNorm(num_groups=1, num_channels=1)
        self.value_linear = nn.Linear(1*3*3, value_layers, bias=False)
        self.value = nn.Linear(value_layers, 1)
        
    def forward(self, x):
        x = F.relu(self.gn_init(self.conv_init(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        policy = F.relu(self.policy_gn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy(policy)
        
        value = F.relu(self.value_gn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_linear(value))
        value = self.value(value)
        value = torch.tanh(value)
        return policy, value