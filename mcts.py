import math
from abc import ABC, abstractmethod

import numpy as np

from game import Game

class MCTSNode:
    def __init__(self, state: Game, prior_prob: float):
        self.state = state
        self.children = {}
        self.visits = 0
        self.t_action_value = 0
        self.m_action_value = 0
        self.prior_prob = prior_prob
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def create_node(self, state, prior_prob):
        return type(self)(state, prior_prob)
    
    @abstractmethod
    def get_model_state(self):
        pass
        
class MCTS:
    def __init__(self, root, temperature: float = 1.0, is_training: bool = True):
        self.root = root
        self.temperature = temperature
        self.is_training = is_training
        self.seen_states = {} # for shared nodes
        
    def get_move(self) -> tuple[int, dict[int, float]]:
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
        
    def select(self) -> tuple[MCTSNode, list[MCTSNode]]:
        node = self.root
        path = [node]
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
            path.append(node)
            
        #step the game and cache the state
        if node is not self.root and node.state is None:
            parent = path[-2]
            node.state = parent.state.copy()
            node.state.step(best_action)
            cache_state = node.state.get_cache_state()
            if cache_state in self.seen_states:
                shared_node = self.seen_states[cache_state]
                parent.children[best_action] = shared_node
                node = shared_node
                path[-1] = shared_node
            else:
                self.seen_states[cache_state] = node
        return node, path
        
    def expand(self, node: MCTSNode, action_probs):
        valid_moves = node.state.get_valid_moves()
        mask = np.array(valid_moves, dtype=bool)
        
        masked_logits = np.copy(action_probs)
        masked_logits[~mask] = -1e9
        
        shift_logits = masked_logits - np.max(masked_logits)
        exp_logits = np.exp(shift_logits)
        probabilities = exp_logits / np.sum(exp_logits)
            
        valid_indicies = np.where(mask)[0]
        for i in valid_indicies:
            child = node.create_node(None, probabilities[i])
            node.children[i] = child
        
    def backpropagate(self, path: list[MCTSNode], value: int):
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            node.visits += 1
            node.t_action_value += value
            node.m_action_value = node.t_action_value / node.visits
            
            if i - 1 >= 0:
                parent = path[i - 1]
                if parent.state is not None and parent.state.turn != node.state.turn:
                    value = -value
            
    def input_move(self, action):
        child = self.root.children[action]
        if child.state is None:
            child.state = child.parent.state.copy()
            child.state.step(action)
        self.root = child
        
    def add_dirlect_noise(self):
        if self.is_training and self.root.children:
            actions = list(self.root.children.keys())
            n_actions = len(actions)
            if n_actions > 0:
                noise = np.random.dirichlet([0.3] * n_actions)
                epsilon = 0.25
                for i, action in enumerate(actions):
                    p = self.root.children[action].prior_prob
                    self.root.children[action].prior_prob = p * (1 - epsilon) + noise[i] * epsilon
        