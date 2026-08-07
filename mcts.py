import math
from abc import ABC, abstractmethod

import numpy as np
import torch

from game import Game, Outcome, Turn
from model import ResNet
        
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
            self.backpropagate(node, value)
        
    def select(self, node):
        while not node.is_leaf():
            argmax = float('-inf')
            best_node = None
            best_action = None
            sqrt_node_visits = math.sqrt(node.visits)
            
            for action, child in node.children.items():
                q = child.m_action_value
                
                if child.state is not None and child.state.turn != node.state.turn:
                    q = -q
                    
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
            node.children[i] = child
        
    def backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.t_action_value += value
            node.m_action_value = node.t_action_value / node.visits
            
            if node.parent is not None and node.parent.state is not None:
                if node.parent.state.turn != node.state.turn:
                    value = -value
            node = node.parent
            
    def input_move(self, action):
        child = self.root.children[action]
        if child.state is None:
            child.state = child.parent.state.copy()
            child.state.step(action)
        self.root = child
        
        
class MCTSNode:
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
    
    @abstractmethod
    def get_model_state(self):
        pass