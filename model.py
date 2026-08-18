import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

class ResBlock(nn.Module):
    def __init__(self, channels=128):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=channels//8, num_channels=channels)
        self.gn2 = nn.GroupNorm(num_groups=channels//8, num_channels=channels)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels=13, n_blocks=10, n_channels=128, value_layers=64, n_actions=8*8*4, board_size=8*8):
        super(ResNet, self).__init__()
        self.conv_init = nn.Conv2d(input_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.gn_init = nn.GroupNorm(num_groups=n_channels//8, num_channels=n_channels)
        
        self.res_blocks = nn.ModuleList([ResBlock(n_channels) for _ in range(n_blocks-1)])
        
        self.policy_conv = nn.Conv2d(n_channels, 2, kernel_size=1, bias=False)
        self.policy_gn = nn.GroupNorm(num_groups=1, num_channels=2)
        self.policy = nn.Linear(2*board_size, n_actions)
        
        self.value_conv = nn.Conv2d(n_channels, 1, kernel_size=1, bias=False)
        self.value_gn = nn.GroupNorm(num_groups=1, num_channels=1)
        self.value_linear = nn.Linear(1*board_size, value_layers, bias=False)
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
    
class ReplayBuffer():
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
            player_k = curr_state[2]
            opponent_k = curr_state[3]
            turn = curr_state[4]
            
            h1_state = game[0][start-1] if start-1 >= 0 else [np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8))]
            h1_player = h1_state[0] if turn[0][0] == h1_state[4][0][0] else np.flip(h1_state[1])
            h1_opponent = h1_state[1] if turn[0][0] == h1_state[4][0][0] else np.flip(h1_state[0])
            h1_player_k = h1_state[2] if turn[0][0] == h1_state[4][0][0] else np.flip(h1_state[3])
            h1_opponent_k = h1_state[3] if turn[0][0] == h1_state[4][0][0] else np.flip(h1_state[2])
            
            h2_state = game[0][start-2] if start-2 >= 0 else [np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8))]
            h2_player = h2_state[0] if turn[0][0] == h2_state[4][0][0] else np.flip(h2_state[1])
            h2_opponent = h2_state[1] if turn[0][0] == h2_state[4][0][0] else np.flip(h2_state[0])
            h2_player_k = h2_state[2] if turn[0][0] == h2_state[4][0][0] else np.flip(h2_state[3])
            h2_opponent_k = h2_state[3] if turn[0][0] == h2_state[4][0][0] else np.flip(h2_state[2])
            
            state = np.stack((player, h1_player, h2_player, opponent, h1_opponent, h2_opponent, player_k, h1_player_k, h2_player_k, opponent_k, h1_opponent_k, h2_opponent_k, turn))
            action_probs.append(curr_action_probs)
            values.append(curr_value)
            states.append(state)
            
        return states, action_probs, values
    
class Trainer():
    def __init__(self, model, device, replay_buffer, lr=0.001, weight_decay=1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.replay_buffer = replay_buffer
        
    def train(self, batch_size):
        self.model.train()
        states, t_action_probs, t_values = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        t_action_probs = torch.FloatTensor(np.array(t_action_probs)).to(self.device)
        t_values = torch.FloatTensor(np.array(t_values)).to(self.device)
        
        self.optimizer.zero_grad()
        p_action_probs, p_values = self.model(states)
        loss = F.cross_entropy(p_action_probs, t_action_probs) + F.mse_loss(p_values.view(-1), t_values.view(-1))
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
                