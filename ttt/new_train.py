import multiprocessing as mp

import torch
import numpy as np

from train import train
from ttt.tictactoe import TicTacToe
from ttt.ttt_replay_buffer import TTTReplayBuffer

def rotate_data(states, probs):
    rotated_data = []
    for k in range(1, 4):
        for state, prob, in zip(states, probs):
            r_states, r_probs = [], []
            p_2d = prob.reshape(3, 3)
            
            r_state = np.rot90(state, k, axes=(1,2))
            r_prob = np.rot90(p_2d, k)
            
            r_states.append(r_state.copy())
            r_probs.append(r_prob.flatten().copy())
        rotated_data.append((r_states, r_probs))

    return rotated_data

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = {
        'game': TicTacToe(),
        'n_games': 10,
        'n_workers': 10,
        'n_iterations': 100,
        'temp_moves': 4,
        'batch_size': 64,
        'mini_batches': 100,
        'training_iterations': 100,
        'train_device': torch.device('cuda'),
        'epsilon': 0.25,
        'alpha': 1.0,
        'worker_device': torch.device('cpu'),
        'model_kwargs': {
            'input_channels': 2,
            'n_blocks': 3,
            'n_channels': 32,
            'value_layers': 16,
            'n_actions': 9,
            'board_size': 3*3
        },
        'replay_buffer': TTTReplayBuffer(max_size=5000),
        'model_folder': './ttt/models/',
        'data_aug_func': rotate_data
    }

    train(**args)