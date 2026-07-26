import math

import numpy as np
import torch

from game import Game, Outcome
from model import ResNet

class MCTSNode:
    def __init__(self, parent, state: Game, prior_prob: float):
        self.parent = parent
        self.state = state
        self.h1_state = None
        self.h2_state = None
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
        player_k = curr_state[2]
        opponent_k = curr_state[3]
        turn = curr_state[4]
            
        h1_state = self.h1_state.get_state() if self.h1_state is not None else [np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8))]
        h1_player = h1_state[0] if turn[0][0] == h1_state[4][0][0] else np.flip(h1_state[1])
        h1_opponent = h1_state[1] if turn[0][0] == h1_state[4][0][0] else np.flip(h1_state[0])
        h1_player_k = h1_state[2] if turn[0][0] == h1_state[4][0][0] else np.flip(h1_state[3])
        h1_opponent_k = h1_state[3] if turn[0][0] == h1_state[4][0][0] else np.flip(h1_state[2])
            
        h2_state = self.h2_state.get_state() if self.h2_state is not None else [np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8))]
        h2_player = h2_state[0] if turn[0][0] == h2_state[4][0][0] else np.flip(h2_state[1])
        h2_opponent = h2_state[1] if turn[0][0] == h2_state[4][0][0] else np.flip(h2_state[0])
        h2_player_k = h2_state[2] if turn[0][0] == h2_state[4][0][0] else np.flip(h2_state[3])
        h2_opponent_k = h2_state[3] if turn[0][0] == h2_state[4][0][0] else np.flip(h2_state[2])
            
        model_state = np.stack((player, h1_player, h2_player, opponent, h1_opponent, h2_opponent, player_k, h1_player_k, h2_player_k, opponent_k, h1_opponent_k, h2_opponent_k, turn))
        return model_state
        
class MCTS:
    def __init__(self, model: ResNet, device, root, iterations: int, temperature: float = 1.0, is_training: bool = True):
        self.model = model
        self.device = device
        self.root = root
        self.temperature = temperature
        self.iterations = iterations
        self.is_training = is_training
        
    def get_move(self):
        self.search()
            
        actions = list(self.root.children.keys())
        visits = np.array([child.visits for child in self.root.children.values()])
        if self.temperature < 0.1:
            max_visits = np.max(visits)
            is_max = visits == max_visits
            probs = is_max / np.sum(is_max)
        else:
            visits = visits ** (1 / self.temperature)
            total_visits = np.sum(visits)
            probs = visits / total_visits
            
        action = np.random.choice(actions, p=probs)
        action_probs = {actions[i]: probs[i] for i in range(len(actions))}
        self.root = self.root.children[action]
        self.root.parent = None
        
        return action, action_probs
        
    def search(self):
        # dirichlet noise
        if not self.root.children:
            model_state = self.root.get_model_state()
            state_tensor = torch.from_numpy(model_state).float().unsqueeze(0)
            state_tensor = state_tensor.to(self.device)
            with torch.no_grad():
                action_probs, _ = self.model(state_tensor)
            action_probs = action_probs[0].detach().cpu().numpy()
            self.expand(self.root, action_probs)
        if self.is_training and self.root.children:
            actions = list(self.root.children.keys())
            n_actions = len(actions)
            if n_actions > 0:
                noise = np.random.dirichlet([0.3] * n_actions)
                epsilon = 0.25
                for i, action in enumerate(actions):
                    p = self.root.children[action].prior_prob
                    self.root.children[action].prior_prob = p * (1 - epsilon) + noise[i] * epsilon
                
        for _ in range(self.iterations):
            # select
            node = self.select(self.root)
            
            # expand
            if node.state.outcome is None:
                model_state = node.get_model_state()
                state_tensor = torch.from_numpy(model_state).float().unsqueeze(0)
                state_tensor = state_tensor.to(self.device)
                with torch.no_grad():
                    action_probs, value = self.model(state_tensor)
                action_probs = action_probs[0].detach().cpu().numpy()
                value = value.item()
                self.expand(node, action_probs)
            else:
                if node.state.outcome == Outcome.DRAW:
                    value = 0
                elif node.state.turn == node.state.outcome:
                    value = 1
                else:
                    value = -1
            
            # backprop
            if node.state.turn != self.root.state.turn:
                value *= -1
            self.backpropagate(node, value)
        
    def select(self, node):
        while not node.is_leaf():
            argmax = float('-inf')
            best_node = None
            best_action = None
            sqrt_node_visits = math.sqrt(node.visits)
            
            for action, child in node.children.items():
                q = child.m_action_value
                    
                u = 1.5 * child.prior_prob * (sqrt_node_visits / (1 + child.visits))
                value = q + u
                
                if value > argmax:
                    argmax = value
                    best_node = child
                    best_action = action
                
            node = best_node
        if node is not self.root and node.state is None:
            node.state = node.parent.state.copy()
            node.state.step(best_action)
        return node
        
    def expand(self, node, action_probs):
        valid_moves = node.state.get_valid_moves()
        mask = np.array(valid_moves, dtype=bool)
        
        masked_logits = np.copy(action_probs)
        masked_logits[~mask] = -1e9
        
        shift_logits = masked_logits - np.max(masked_logits) # numeric stability
        exp_logits = np.exp(shift_logits)
        probabilities = exp_logits / np.sum(exp_logits)
            
        valid_indicies = np.where(mask)[0]
        for i in valid_indicies:
            child = node.create_node(node, None, probabilities[i])
            child.h1_state = node.state
            child.h2_state = node.h1_state
            node.children[i] = child
        
    def backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            if self.root.state.turn == node.state.turn:
                node.t_action_value += value
            else:
                node.t_action_value -= value
            node.m_action_value = node.t_action_value / node.visits
            node = node.parent
            
    def input_move(self, action):
        child = self.root.children[action]
        if child.state is None:
            child.state = child.parent.state.copy()
            child.state.step(action)
        self.root = child
        
            
class TTTMCTSNode:
    def __init__(self, parent, state: Game, prior_prob: float):
        self.parent = parent
        self.state = state
        self.h1_state = None
        self.h2_state = None
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
        turn = curr_state[2]
            
        h1_state = self.h1_state.get_state() if self.h1_state is not None else [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]
        h1_player = h1_state[0] if turn[0][0] == h1_state[2][0][0] else h1_state[1]
        h1_opponent = h1_state[1] if turn[0][0] == h1_state[2][0][0] else h1_state[0]
            
        h2_state = self.h2_state.get_state() if self.h2_state is not None else [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]
        h2_player = h2_state[0] if turn[0][0] == h2_state[2][0][0] else h2_state[1]
        h2_opponent = h2_state[1] if turn[0][0] == h2_state[2][0][0] else h2_state[0]
            
        model_state = np.stack((player, h1_player, h2_player, opponent, h1_opponent, h2_opponent, turn))
        return model_state